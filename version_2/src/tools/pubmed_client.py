"""
PubMed / NCBI E-utilities client for Drug Repurposing v2.

Provides:
  - search_pubmed(query, max_results) → list of PMIDs
  - build_repurposing_query(drug, disease) → optimized PubMed query string
  - fetch_abstracts(pmids) → list of article dicts

Used by Agent 4 (Literature RAG).
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

from loguru import logger

from src.config import get_settings
from src.tools.http_utils import APIError, get_json, get_text

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _rate_delay() -> None:
    """Respect NCBI rate limits: 10 req/s with key, 3 req/s without."""
    settings = get_settings()
    time.sleep(0.11 if settings.ncbi_api_key else 0.34)


def build_repurposing_query(drug_name: str, disease_name: str, is_cns: bool = False) -> str:
    """
    Build an optimized PubMed query for a drug-disease repurposing pair.
    Boosts for clinical trials, systematic reviews, and mechanistic studies.
    """
    base = (
        f'("{drug_name}"[Title/Abstract] OR "{drug_name}"[MeSH Terms])'
        f' AND ("{disease_name}"[Title/Abstract] OR "{disease_name}"[MeSH Terms])'
        f' AND ("repurposing"[Title/Abstract] OR "repositioning"[Title/Abstract]'
        f' OR "treatment"[Title/Abstract] OR "mechanism"[Title/Abstract]'
        f' OR "therapeutic"[Title/Abstract] OR "clinical trial"[Title/Abstract])'
    )
    if is_cns:
        base += (
            ' AND ("neuroprotection"[Title/Abstract] OR "cognitive"[Title/Abstract]'
            ' OR "neurodegeneration"[Title/Abstract] OR "brain"[Title/Abstract])'
        )
    return base


def search_pubmed(
    query: str,
    max_results: int = 15,
    min_year: int = 2018,
) -> list[str]:
    """
    Search PubMed with E-utilities esearch.

    Args:
        query: PubMed query string
        max_results: Maximum number of PMIDs to return
        min_year: Earliest publication year

    Returns:
        List of PMID strings
    """
    _rate_delay()
    settings = get_settings()

    if min_year:
        query += f" AND {min_year}:{datetime.now().year + 1}[pdat]"

    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
    }
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key

    try:
        data = get_json(_ESEARCH, params=params)
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.debug(f"PubMed esearch returned {len(pmids)} PMIDs")
        return pmids
    except APIError as e:
        logger.warning(f"PubMed search failed: {e}")
        return []


def fetch_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    """
    Fetch full abstracts for a list of PMIDs using efetch.

    Returns:
        List of dicts: {pmid, title, abstract, year, url, journal}
    """
    if not pmids:
        return []

    _rate_delay()
    settings = get_settings()

    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key

    try:
        xml_text = get_text(_EFETCH, params=params)
        return _parse_pubmed_xml(xml_text)
    except APIError as e:
        logger.warning(f"PubMed efetch failed: {e}")
        return []


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed efetch XML into structured article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"PubMed XML parse error: {e}")
        return []

    for pub in root.findall(".//PubmedArticle"):
        try:
            med = pub.find("MedlineCitation")
            if med is None:
                continue

            pmid_el = med.find("PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

            art = med.find("Article")
            if art is None:
                continue

            # Title
            title_el = art.find("ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstract (may have multiple sections)
            abstract_parts = art.findall("Abstract/AbstractText")
            abstract = " ".join("".join(p.itertext()).strip() for p in abstract_parts)

            # Journal & year
            journal = art.findtext("Journal/Title", default="")
            year = art.findtext("Journal/JournalIssue/PubDate/Year", default="")
            if not year:
                # Try MedlineDate fallback
                medline_date = art.findtext("Journal/JournalIssue/PubDate/MedlineDate", default="")
                year = medline_date[:4] if medline_date else ""

            if not pmid or not abstract:
                continue

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        except Exception as e:
            logger.debug(f"Skipping malformed PubMed record: {e}")
            continue

    logger.debug(f"Parsed {len(articles)} articles from PubMed XML")
    return articles


def get_recent_papers_count(drug_name: str, disease_name: str) -> int:
    """Quick count of recent papers for a drug-disease pair (last 7 years)."""
    query = build_repurposing_query(drug_name, disease_name)
    pmids = search_pubmed(query, max_results=50)
    return len(pmids)
