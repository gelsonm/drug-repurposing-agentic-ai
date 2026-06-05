"""
PubMed / NCBI E-utilities client.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
No key: 3 req/s | With NCBI_API_KEY: 10 req/s
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any


from loguru import logger

from src.config import get_settings
from src.tools.http_utils import APIError, get_json, get_text

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _rate_delay() -> None:
    """Respect NCBI rate limits."""
    settings = get_settings()
    delay = 0.11 if settings.ncbi_api_key else 0.34  # 10/s vs 3/s
    time.sleep(delay)


def _base_params() -> dict[str, str]:
    """Base params for all E-utilities requests."""
    settings = get_settings()
    params: dict[str, str] = {"retmode": "json"}
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return params


def search_pubmed(
    query: str,
    max_results: int = 20,
    sort: str = "relevance",
    min_year: int | None = None,
) -> list[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed query string. Supports MeSH terms, field tags, etc.
               e.g. "metformin[MeSH] AND Alzheimer disease[MeSH]"
        max_results: Maximum number of results to return.
        sort: Sort order: "relevance" or "pub_date".
        min_year: If set, restrict results to >= this publication year.

    Returns:
        List of PMID strings.
    """
    if min_year:
        query = f"{query} AND {min_year}:{datetime.now().year + 1}[pdat]"

    logger.info(f"PubMed: searching '{query}' (max {max_results})")
    _rate_delay()

    params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "sort": sort,
        "usehistory": "n",
    }

    try:
        data = get_json(_ESEARCH_URL, params=params)
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"PubMed: found {len(pmids)} results")
        return pmids
    except APIError as e:
        logger.warning(f"PubMed search failed: {e}")
        return []


def fetch_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    """
    Fetch article metadata and abstracts for a list of PMIDs.

    Returns list of dicts with: pmid, title, abstract, authors,
    journal, year, doi, url.
    """
    if not pmids:
        return []

    logger.info(f"PubMed: fetching {len(pmids)} abstracts")
    _rate_delay()

    # Use efetch to get XML (richer than JSON for abstracts)
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if get_settings().ncbi_api_key:
        params["api_key"] = get_settings().ncbi_api_key

    try:
        xml_text = get_text(_EFETCH_URL, params=params)
        return _parse_pubmed_xml(xml_text)
    except APIError as e:
        logger.warning(f"PubMed fetch failed: {e}")
        return []


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed XML response into structured dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find("MedlineCitation")
                if medline is None:
                    continue

                # PMID
                pmid_el = medline.find("PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                art = medline.find("Article")
                if art is None:
                    continue

                # Title
                title_el = art.find("ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""

                # Abstract
                abstract_el = art.find("Abstract/AbstractText")
                if abstract_el is not None:
                    abstract = "".join(abstract_el.itertext())
                else:
                    # Handle structured abstracts
                    abstract_parts = art.findall("Abstract/AbstractText")
                    abstract = " ".join("".join(p.itertext()) for p in abstract_parts)

                # Journal and year
                journal = art.findtext("Journal/Title", default="")
                year = art.findtext("Journal/JournalIssue/PubDate/Year", default="")

                # Authors
                authors = []
                for author in art.findall("AuthorList/Author"):
                    last = author.findtext("LastName", "")
                    initials = author.findtext("Initials", "")
                    if last:
                        authors.append(f"{last} {initials}".strip())

                # DOI
                doi = ""
                for id_el in article.findall(".//ArticleId"):
                    if id_el.get("IdType") == "doi":
                        doi = id_el.text or ""

                articles.append({
                    "pmid": pmid,
                    "title": title.strip(),
                    "abstract": abstract.strip(),
                    "authors": authors[:5],  # First 5 authors
                    "journal": journal,
                    "year": year,
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            except Exception as e:
                logger.debug(f"Error parsing article: {e}")
                continue
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")

    return articles


def search_and_fetch(
    query: str,
    max_results: int = 10,
    min_year: int | None = 2020,
) -> list[dict[str, Any]]:
    """
    Convenience function: search PubMed and fetch abstracts in one call.
    Returns list of article metadata dicts.
    """
    pmids = search_pubmed(query, max_results=max_results, min_year=min_year)
    if not pmids:
        return []
    return fetch_abstracts(pmids)


def build_repurposing_query(drug_name: str, disease_name: str) -> str:
    """Build an optimized PubMed query for drug-disease repurposing evidence."""
    return (
        f'("{drug_name}"[Title/Abstract] OR "{drug_name}"[MeSH Terms]) '
        f'AND ("{disease_name}"[Title/Abstract] OR "{disease_name}"[MeSH Terms]) '
        f'AND ("drug repurposing"[Title/Abstract] OR "drug repositioning"[Title/Abstract] '
        f'OR "therapeutic"[Title/Abstract] OR "treatment"[Title/Abstract] '
        f'OR "clinical trial"[Title/Abstract])'
    )


def build_cancer_repurposing_query(drug_name: str, cancer_name: str) -> str:
    """
    Build an oncology-optimized PubMed query for cancer drug repurposing.
    Uses cancer-specific MeSH terms and evidence filters.
    """
    return (
        f'("{drug_name}"[Title/Abstract] OR "{drug_name}"[MeSH Terms]) '
        f'AND ("{cancer_name}"[Title/Abstract] OR "{cancer_name}"[MeSH Terms] '
        f'OR "neoplasm"[MeSH Terms] OR "tumor"[Title/Abstract]) '
        f'AND ("antineoplastic"[Title/Abstract] OR "anticancer"[Title/Abstract] '
        f'OR "repurposing"[Title/Abstract] OR "repositioning"[Title/Abstract] '
        f'OR "clinical trial"[Publication Type] OR "in vivo"[Title/Abstract] '
        f'OR "apoptosis"[Title/Abstract] OR "tumor suppression"[Title/Abstract])'
    )

