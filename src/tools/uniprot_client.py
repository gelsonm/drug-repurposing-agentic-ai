"""
UniProt REST API client.
Docs: https://www.uniprot.org/help/api
No authentication required. Generous rate limits.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.tools.http_utils import APIError, get_json, get_text

_BASE_URL = "https://rest.uniprot.org/uniprotkb"
_RATE_LIMIT_DELAY = 0.3


def get_protein_by_accession(accession: str) -> dict[str, Any] | None:
    """
    Fetch full UniProt entry for a protein accession (e.g. 'P00533').

    Returns structured dict with: protein name, function, disease associations,
    gene names, organism, sequence length, keywords.
    """
    logger.info(f"UniProt: fetching protein {accession}")
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        data = get_json(f"{_BASE_URL}/{accession}.json")
        return data
    except APIError as e:
        logger.warning(f"UniProt: failed to fetch {accession}: {e}")
        return None


def search_proteins(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """
    Search UniProt by keyword, gene name, or disease.

    Example queries:
    - "AMPK human" -> AMPK protein in Homo sapiens
    - "gene:PRKAA1 AND organism_id:9606"
    - "disease:Alzheimer AND reviewed:true"
    """
    logger.info(f"UniProt: searching '{query}'")
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        data = get_json(
            f"{_BASE_URL}/search",
            params={
                "query": query,
                "format": "json",
                "size": max_results,
                "fields": "accession,id,protein_name,gene_names,organism_name,function,disease,keyword",
            },
        )
        results = data.get("results", [])
        logger.info(f"UniProt: found {len(results)} proteins for '{query}'")
        return results
    except APIError as e:
        logger.warning(f"UniProt search failed for '{query}': {e}")
        return []


def extract_protein_function(entry: dict[str, Any]) -> str:
    """Extract plain-text function description from UniProt entry."""
    try:
        comments = entry.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    return texts[0].get("value", "")
    except Exception:
        pass
    return ""


def extract_disease_associations(entry: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extract disease associations from UniProt entry.
    Returns list of dicts with: disease_name, description, evidence.
    """
    diseases = []
    try:
        comments = entry.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "DISEASE":
                disease = comment.get("disease", {})
                diseases.append({
                    "disease_name": disease.get("diseaseId", ""),
                    "description": disease.get("description", ""),
                    "acronym": disease.get("acronym", ""),
                })
    except Exception:
        pass
    return diseases


def extract_gene_names(entry: dict[str, Any]) -> list[str]:
    """Extract all gene names from UniProt entry."""
    genes = []
    try:
        gene_list = entry.get("genes", [])
        for gene in gene_list:
            gn = gene.get("geneName", {}).get("value", "")
            if gn:
                genes.append(gn)
            for syn in gene.get("synonyms", []):
                if syn.get("value"):
                    genes.append(syn["value"])
    except Exception:
        pass
    return list(set(genes))


def get_proteins_for_targets(uniprot_ids: list[str]) -> list[dict[str, Any]]:
    """Batch fetch protein metadata for a list of UniProt accessions."""
    results = []
    for uid in uniprot_ids[:10]:  # cap to avoid overloading
        entry = get_protein_by_accession(uid)
        if entry:
            results.append(entry)
        time.sleep(_RATE_LIMIT_DELAY)
    return results
