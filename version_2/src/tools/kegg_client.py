"""
KEGG REST API client.
Documentation: https://www.genome.jp/kegg/rest/

Used by Agent 5 (Pathway Validator) to:
- Map gene symbols to KEGG pathway IDs
- Retrieve pathway descriptions and gene members
- Check if a drug's targets sit within a disease pathway
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.tools.http_utils import APIError, get_text

_BASE = "https://rest.kegg.jp"
_RATE_DELAY = 0.3


def _parse_list(text: str) -> list[tuple[str, str]]:
    """Parse KEGG tab-delimited list response into (id, name) pairs."""
    results = []
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            results.append((parts[0].strip(), parts[1].strip()))
    return results


def get_pathways_for_gene(gene_symbol: str, organism: str = "hsa") -> list[dict[str, Any]]:
    """
    Get KEGG pathways containing a given gene symbol.

    Args:
        gene_symbol: Human gene symbol (e.g. "MTOR", "AMPK", "BACE1")
        organism: KEGG organism code (default "hsa" = human)

    Returns:
        List of {pathway_id, pathway_name} dicts
    """
    logger.info(f"KEGG: finding pathways for gene '{gene_symbol}'")
    time.sleep(_RATE_DELAY)

    try:
        text = get_text(f"{_BASE}/find/pathway/{gene_symbol}")
        if not text or text.strip() == "":
            # Try searching by organism-prefixed gene
            time.sleep(_RATE_DELAY)
            text = get_text(f"{_BASE}/link/pathway/{organism}:{gene_symbol}")

        pathways = []
        for line in text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                raw_id = parts[0].strip().replace("path:", "")
                raw_name = parts[1].strip()
                # Only include human pathways
                if raw_id.startswith("hsa") or raw_id.startswith("map"):
                    pathways.append({"pathway_id": raw_id, "pathway_name": raw_name})

        logger.info(f"KEGG: {len(pathways)} pathways found for gene '{gene_symbol}'")
        return pathways[:10]

    except APIError as e:
        logger.warning(f"KEGG gene pathway lookup failed for '{gene_symbol}': {e}")
        return []


def get_genes_in_pathway(pathway_id: str) -> list[str]:
    """
    Get all genes in a KEGG pathway.

    Args:
        pathway_id: KEGG pathway ID (e.g. "hsa04150")

    Returns:
        List of gene symbols
    """
    logger.info(f"KEGG: getting genes in pathway {pathway_id}")
    time.sleep(_RATE_DELAY)

    clean_id = pathway_id.replace("path:", "")
    try:
        text = get_text(f"{_BASE}/link/gene/{clean_id}")
        genes = []
        for line in text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                gene_id = parts[1].replace("hsa:", "").strip()
                if gene_id:
                    genes.append(gene_id)
        return genes
    except APIError as e:
        logger.warning(f"KEGG pathway gene lookup failed for {pathway_id}: {e}")
        return []


def search_pathways_by_keyword(keyword: str) -> list[dict[str, Any]]:
    """
    Search KEGG for pathways matching a keyword (e.g. "mTOR", "tau", "amyloid").

    Returns list of {pathway_id, pathway_name}.
    """
    logger.info(f"KEGG: searching pathways for keyword '{keyword}'")
    time.sleep(_RATE_DELAY)

    try:
        text = get_text(f"{_BASE}/find/pathway/{keyword}")
        if not text.strip():
            return []
        pairs = _parse_list(text)
        results = []
        for pid, pname in pairs:
            clean_pid = pid.replace("path:", "")
            if clean_pid.startswith("hsa") or clean_pid.startswith("map"):
                results.append({"pathway_id": clean_pid, "pathway_name": pname})
        logger.info(f"KEGG: {len(results)} pathways matched '{keyword}'")
        return results[:10]
    except APIError as e:
        logger.warning(f"KEGG pathway search failed for '{keyword}': {e}")
        return []


def get_drug_targets_from_kegg(drug_kegg_id: str) -> list[str]:
    """
    Get protein targets of a drug from KEGG Drug database.
    Drug IDs are D##### format (e.g. "D07867" for Metformin).
    """
    time.sleep(_RATE_DELAY)
    try:
        text = get_text(f"{_BASE}/link/target/{drug_kegg_id}")
        targets = []
        for line in text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                targets.append(parts[1].strip())
        return targets
    except APIError:
        return []


# ── Disease pathway mapping (curated, for reliable fallback) ──────────────

DISEASE_PATHWAY_MAP: dict[str, list[str]] = {
    "alzheimer": ["hsa05010", "hsa04150", "hsa04210", "hsa04722"],
    "alzheimer's disease": ["hsa05010", "hsa04150", "hsa04210", "hsa04722"],
    "parkinson": ["hsa05012", "hsa04141", "hsa04210"],
    "parkinson's disease": ["hsa05012", "hsa04141", "hsa04210"],
    "colorectal cancer": ["hsa05210", "hsa04310", "hsa04020"],
    "breast cancer": ["hsa05224", "hsa04151", "hsa04010"],
    "pancreatic cancer": ["hsa05212", "hsa04151", "hsa04010"],
    "type 2 diabetes": ["hsa04930", "hsa04150", "hsa04910"],
    "glioblastoma": ["hsa05214", "hsa04151", "hsa04010"],
    "multiple myeloma": ["hsa05216", "hsa04151"],
}


def get_disease_pathways_curated(disease_name: str) -> list[str]:
    """Return curated KEGG pathway IDs for a disease (fast fallback)."""
    key = disease_name.lower()
    for k, pathways in DISEASE_PATHWAY_MAP.items():
        if k in key or key in k:
            return pathways
    return []
