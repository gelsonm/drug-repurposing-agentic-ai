"""
Hallucination guard for drug repurposing v2.

Every factual claim from an LLM agent must be anchored to at least one
verifiable source ID before being passed downstream.

Supported ID formats:
  - PMID: numeric string (e.g. 38521091)
  - ChEMBL ID: CHEMBL followed by digits (e.g. CHEMBL1431)
  - DrugBank ID: DB##### format (e.g. DB00331)
  - KEGG pathway: hsa##### or map##### (e.g. hsa04150)
  - Reactome ID: R-HSA-#### format (e.g. R-HSA-165159)
  - UniProt: 6-character alphanumeric (e.g. P37231)
  - Open Targets EFO: EFO_#### format (e.g. EFO_0000249)
  - OMIM: 6-digit number (e.g. 104300)
  - ClinicalTrials: NCT followed by 8 digits (e.g. NCT04777396)
"""
from __future__ import annotations

import re
from loguru import logger

# Regex patterns for known DB identifier formats
_PATTERNS: dict[str, re.Pattern] = {
    "pubmed": re.compile(r"^\d{7,8}$"),
    "chembl": re.compile(r"^CHEMBL\d+$", re.IGNORECASE),
    "drugbank": re.compile(r"^DB\d{5}$", re.IGNORECASE),
    "kegg": re.compile(r"^(hsa|map)\d{5}$"),
    "reactome": re.compile(r"^R-HSA-\d+$"),
    "uniprot": re.compile(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$"),
    "efo": re.compile(r"^EFO_\d+$"),
    "omim": re.compile(r"^\d{6}$"),
    "clinicaltrials": re.compile(r"^NCT\d{8}$", re.IGNORECASE),
    "string": re.compile(r"^\d+\.\w+$"),  # STRING: e.g. "9606.ENSP..."
    "disgenet": re.compile(r"^C\d+$"),    # UMLS CUI format used by DisGeNET
}


def detect_id_type(source_id: str) -> str | None:
    """Return the database type for a known ID format, or None if unrecognized."""
    sid = source_id.strip()
    for db_type, pattern in _PATTERNS.items():
        if pattern.match(sid):
            return db_type
    return None


def is_valid_db_id(source_id: str) -> bool:
    """Return True if the ID matches a recognized database format."""
    return detect_id_type(source_id) is not None


def validate_claim(claim: str, source_ids: list[str]) -> bool:
    """
    Validate that a factual claim has at least one verifiable source ID.

    Args:
        claim: The text claim being made by an LLM agent.
        source_ids: List of database IDs cited for this claim.

    Returns:
        True if the claim is grounded; False if it should be rejected.
    """
    if not source_ids:
        logger.warning(f"HALLUCINATION GUARD: Unsourced claim rejected: '{claim[:80]}...'")
        return False

    valid = [sid for sid in source_ids if is_valid_db_id(sid)]
    if not valid:
        logger.warning(
            f"HALLUCINATION GUARD: No valid DB IDs in {source_ids} for claim: '{claim[:60]}'"
        )
        return False

    return True


def validate_candidate_sources(drug_name: str, source_ids: list[str]) -> dict:
    """
    Validate all sources cited for a drug candidate.

    Returns:
        dict with 'valid', 'invalid', 'has_pubmed', 'has_kg', 'pass'
    """
    valid = []
    invalid = []
    for sid in source_ids:
        if is_valid_db_id(sid):
            valid.append(sid)
        else:
            invalid.append(sid)

    has_pubmed = any(detect_id_type(s) == "pubmed" for s in valid)
    has_kg = any(detect_id_type(s) in ("chembl", "drugbank", "kegg", "reactome") for s in valid)

    result = {
        "drug": drug_name,
        "valid_ids": valid,
        "invalid_ids": invalid,
        "has_pubmed": has_pubmed,
        "has_kg_source": has_kg,
        "pass": len(valid) >= 1,
    }

    if not result["pass"]:
        logger.warning(f"HALLUCINATION GUARD: Candidate '{drug_name}' has no valid sources.")
    else:
        logger.debug(f"HALLUCINATION GUARD: '{drug_name}' OK — {len(valid)} valid IDs")

    return result


def extract_ids_from_text(text: str) -> list[str]:
    """
    Extract all database IDs embedded in a block of text.
    Useful for post-processing LLM outputs.
    """
    found = []
    # Find all word-like tokens
    tokens = re.findall(r'\b[\w\-\.]+\b', text)
    for token in tokens:
        if is_valid_db_id(token):
            found.append(token)
    return list(set(found))
