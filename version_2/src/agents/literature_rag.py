"""
Agent 4: Literature RAG Agent

Hybrid BM25 + dense (ChromaDB) retrieval over PubMed abstracts.
Distinguishes: positive / negative / mechanistic-only evidence.

Validation: Every claim tied to a PMID. Flags low-evidence candidates.
"""
from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

from src.config import get_settings
from src.schemas.models import CandidateDrug
from src.tools.http_utils import APIError, get_json, get_text

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_COLLECTION_NAME = "repurposing_v2_literature"
_client: chromadb.PersistentClient | None = None
_collection = None

# Evidence signal keywords
_POSITIVE_KEYWORDS = [
    "repurposing", "effective", "beneficial", "protective", "reduced", "improves",
    "significant", "positive", "promising", "successful", "therapeutic benefit",
    "neuroprotective", "anti-tumor", "anti-cancer",
]
_NEGATIVE_KEYWORDS = [
    "failed", "no effect", "ineffective", "not significant", "withdrew",
    "terminated", "adverse", "toxic", "worsened", "contraindicated",
]


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    settings = get_settings()
    persist_dir = str(settings.chroma_db_path)
    _client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB v2 collection ready: {_collection.count()} documents")
    return _collection


def _rate_delay():
    settings = get_settings()
    time.sleep(0.11 if settings.ncbi_api_key else 0.34)


def _build_query(drug_name: str, disease_name: str, is_cns: bool = False) -> str:
    """Build cancer/CNS/general-optimized PubMed query."""
    base = (
        f'("{drug_name}"[Title/Abstract] OR "{drug_name}"[MeSH Terms]) '
        f'AND ("{disease_name}"[Title/Abstract] OR "{disease_name}"[MeSH Terms]) '
        f'AND ("repurposing"[Title/Abstract] OR "repositioning"[Title/Abstract] '
        f'OR "treatment"[Title/Abstract] OR "therapeutic"[Title/Abstract] '
        f'OR "clinical trial"[Title/Abstract])'
    )
    if is_cns:
        base += ' AND ("neuroprotection"[Title/Abstract] OR "cognitive"[Title/Abstract] OR "neurodegeneration"[Title/Abstract])'
    return base


def _search_pubmed(query: str, max_results: int = 15, min_year: int = 2018) -> list[str]:
    if min_year:
        query += f" AND {min_year}:{datetime.now().year + 1}[pdat]"
    _rate_delay()
    settings = get_settings()
    params: dict = {"db": "pubmed", "term": query, "retmax": max_results, "sort": "relevance", "retmode": "json"}
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    try:
        data = get_json(_ESEARCH, params=params)
        return data.get("esearchresult", {}).get("idlist", [])
    except APIError as e:
        logger.warning(f"PubMed search failed: {e}")
        return []


def _fetch_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    if not pmids:
        return []
    _rate_delay()
    settings = get_settings()
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    try:
        xml_text = get_text(_EFETCH, params=params)
        return _parse_xml(xml_text)
    except APIError as e:
        logger.warning(f"PubMed fetch failed: {e}")
        return []


def _parse_xml(xml_text: str) -> list[dict[str, Any]]:
    articles = []
    try:
        root = ET.fromstring(xml_text)
        for pub in root.findall(".//PubmedArticle"):
            try:
                med = pub.find("MedlineCitation")
                if med is None:
                    continue
                pmid = (med.find("PMID") or ET.Element("")).text or ""
                art = med.find("Article")
                if art is None:
                    continue
                title = "".join((art.find("ArticleTitle") or ET.Element("")).itertext())
                abstract_parts = art.findall("Abstract/AbstractText")
                abstract = " ".join("".join(p.itertext()) for p in abstract_parts)
                year = art.findtext("Journal/JournalIssue/PubDate/Year", "")
                articles.append({
                    "pmid": pmid, "title": title.strip(),
                    "abstract": abstract.strip(), "year": year,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            except Exception:
                continue
    except ET.ParseError:
        pass
    return articles


def _classify_evidence(text: str) -> str:
    """Classify abstract as positive/negative/mechanistic."""
    text_lower = text.lower()
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    if pos > neg and pos >= 2:
        return "positive"
    if neg > pos and neg >= 2:
        return "negative"
    return "mechanistic"


def _bm25_score(query_terms: list[str], text: str) -> float:
    """Simple BM25-like term frequency score."""
    try:
        from rank_bm25 import BM25Okapi
        tokenized = text.lower().split()
        bm25 = BM25Okapi([tokenized])
        scores = bm25.get_scores(query_terms)
        return float(scores[0]) if len(scores) > 0 else 0.0
    except ImportError:
        # Fallback: simple TF
        text_lower = text.lower()
        return sum(1.0 for t in query_terms if t in text_lower) / max(1, len(query_terms))


def _ingest_articles(articles: list[dict[str, Any]]) -> int:
    collection = _get_collection()
    settings = get_settings()
    docs, ids, metas = [], [], []

    for article in articles:
        text = f"{article.get('title', '')}. {article.get('abstract', '')}"
        if len(text) < 50:
            continue
        words = text.split()
        chunks = []
        step = settings.rag_chunk_size - settings.rag_chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + settings.rag_chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        for i, chunk in enumerate(chunks):
            doc_id = f"pmid{article.get('pmid', 'x')}_{hashlib.md5(chunk.encode()).hexdigest()[:10]}_{i}"
            docs.append(chunk)
            ids.append(doc_id)
            metas.append({
                "pmid": str(article.get("pmid", "")),
                "title": article.get("title", "")[:200],
                "year": str(article.get("year", "")),
                "url": article.get("url", ""),
                "evidence_type": _classify_evidence(text),
            })

    if not docs:
        return 0
    try:
        collection.upsert(documents=docs, ids=ids, metadatas=metas)
        return len(docs)
    except Exception as e:
        logger.warning(f"ChromaDB upsert failed: {e}")
        return 0


def _query_rag(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Hybrid BM25 + dense retrieval."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    # Dense retrieval
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k * 2, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense_passages = []
        for i, doc in enumerate(results["documents"][0]):
            dense_score = max(0.0, 1.0 - results["distances"][0][i])
            meta = results["metadatas"][0][i]
            dense_passages.append({
                "text": doc, "dense_score": dense_score,
                "pmid": meta.get("pmid", ""), "title": meta.get("title", ""),
                "year": meta.get("year", ""), "url": meta.get("url", ""),
                "evidence_type": meta.get("evidence_type", "mechanistic"),
            })
    except Exception as e:
        logger.warning(f"Dense retrieval failed: {e}")
        dense_passages = []

    # BM25 re-ranking
    query_terms = [t.lower() for t in query.split() if len(t) > 3]
    for p in dense_passages:
        bm25_s = _bm25_score(query_terms, p["text"])
        # Normalize BM25 to [0, 1]
        p["bm25_score"] = min(1.0, bm25_s / max(1.0, bm25_s))
        settings = get_settings()
        p["relevance_score"] = round(
            settings.rag_dense_weight * p["dense_score"]
            + settings.rag_bm25_weight * p["bm25_score"],
            4,
        )

    sorted_passages = sorted(dense_passages, key=lambda p: p["relevance_score"], reverse=True)
    return sorted_passages[:top_k]


def run_literature_rag_agent(
    candidate: CandidateDrug,
    disease_name: str,
    is_cns: bool = False,
    max_papers: int = 15,
) -> tuple[CandidateDrug, list[dict[str, Any]]]:
    """
    Execute Agent 4: Literature RAG.

    Returns:
        Tuple of (enriched CandidateDrug, list of passage dicts)
    """
    drug_name = candidate.drug_name
    logger.info(f"[Agent 4] Literature RAG: {drug_name} + {disease_name}")

    query = _build_query(drug_name, disease_name, is_cns)
    pmids = _search_pubmed(query, max_results=max_papers)
    logger.info(f"[Agent 4] PubMed: {len(pmids)} PMIDs found")

    articles = []
    if pmids:
        articles = _fetch_abstracts(pmids)
        _ingest_articles(articles)

    # Classify evidence
    positive_count = 0
    negative_count = 0
    for art in articles:
        text = f"{art.get('title', '')} {art.get('abstract', '')}"
        label = _classify_evidence(text)
        if label == "positive":
            positive_count += 1
        elif label == "negative":
            negative_count += 1

    # RAG query
    rag_query = f"{drug_name} treatment mechanism {disease_name} efficacy"
    passages = _query_rag(rag_query, top_k=10)

    # Update candidate
    candidate.literature_pmids = pmids[:15]
    candidate.literature_positive_count = positive_count
    candidate.literature_negative_count = negative_count

    if len(pmids) < 3:
        logger.warning(f"[Agent 4] LOW-EVIDENCE FLAG: only {len(pmids)} papers for {drug_name}+{disease_name}")

    logger.info(
        f"[Agent 4] {drug_name}: {len(pmids)} papers, "
        f"+{positive_count}/-{negative_count} evidence, "
        f"{len(passages)} RAG passages"
    )
    return candidate, passages
