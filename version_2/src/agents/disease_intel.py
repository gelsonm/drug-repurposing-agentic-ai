"""
Agent 1: Disease Intelligence Agent
Characterizes the target disease in structured biological terms.

Sources:
  - Open Targets Platform (primary) — disease EFO ID, gene associations, known drugs
  - DisGeNET (supplementary, if API key set) — curated GDA scores
  - PubMed — recent mechanistic papers (via pubmed_client)
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.schemas.models import DiseaseProfile, GeneAssociation
from src.tools.disgenet_client import get_disease_gene_associations
from src.tools.open_targets_client import get_disease_info, get_disease_targets
from src.tools.pubmed_client import build_repurposing_query, search_pubmed

# CNS disease keywords — triggers BBB check downstream
_CNS_KEYWORDS = [
    "alzheimer", "parkinson", "huntington", "als", "epilepsy", "schizophrenia",
    "depression", "bipolar", "glioblastoma", "brain", "neurological", "dementia",
    "multiple sclerosis", "stroke", "migraine", "autism",
]


def run_disease_intel_agent(disease_name: str) -> DiseaseProfile:
    """
    Execute Agent 1: Disease Intelligence.

    Args:
        disease_name: Free-text disease name (e.g., "Alzheimer's disease")

    Returns:
        DiseaseProfile with structured biological characterization
    """
    logger.info(f"[Agent 1] Disease Intelligence: characterizing '{disease_name}'")
    data_sources = []

    # ── Step 1: Open Targets — disease lookup ─────────────────────────────
    efo_id = None
    mondo_id = None
    ot_disease_info = get_disease_info(disease_name)

    if ot_disease_info:
        efo_id = ot_disease_info.get("id")  # EFO ID
        data_sources.append(f"OpenTargets:{efo_id}")
        logger.info(f"[Agent 1] EFO ID: {efo_id}")

    # ── Step 2: Open Targets — gene-disease associations ──────────────────
    ot_targets: list[GeneAssociation] = []
    top_gene_symbols: list[str] = []

    if efo_id:
        raw_targets = get_disease_targets(efo_id, max_results=20)
        for t in raw_targets:
            ot_targets.append(GeneAssociation(
                gene_symbol=t.get("symbol", ""),
                gene_id=t.get("target_id"),
                association_score=round(float(t.get("association_score", 0.0)), 4),
                source="Open Targets",
            ))
        top_gene_symbols = [t.gene_symbol for t in ot_targets[:10] if t.gene_symbol]

    # ── Step 3: DisGeNET — supplementary associations ─────────────────────
    disgenet_genes = get_disease_gene_associations(disease_name, max_results=15)
    if disgenet_genes:
        data_sources.append("DisGeNET:v7.0")
        # Merge: add DisGeNET genes not already in OT list
        ot_symbols = {t.gene_symbol for t in ot_targets}
        for g in disgenet_genes:
            if g["gene_symbol"] and g["gene_symbol"] not in ot_symbols:
                ot_targets.append(GeneAssociation(
                    gene_symbol=g["gene_symbol"],
                    gene_id=g.get("gene_id"),
                    association_score=g["score"],
                    source="DisGeNET",
                ))
                top_gene_symbols.append(g["gene_symbol"])

    # ── Step 4: Known pathways (curated + PubMed context) ─────────────────
    key_pathways = _infer_pathways(disease_name, top_gene_symbols)

    # Quick PubMed count for biological context
    pubmed_query = f'"{disease_name}"[Title/Abstract] AND "pathophysiology"[Title/Abstract]'
    pmid_count = len(search_pubmed(pubmed_query, max_results=5))
    if pmid_count > 0:
        data_sources.append(f"PubMed:{pmid_count}_recent_papers")

    # ── Step 5: CNS flag ──────────────────────────────────────────────────
    is_cns = any(kw in disease_name.lower() for kw in _CNS_KEYWORDS)

    # ── Build profile ─────────────────────────────────────────────────────
    profile = DiseaseProfile(
        disease_name=disease_name,
        efo_id=efo_id,
        key_pathways=key_pathways,
        known_targets=ot_targets[:15],
        disease_genes=list(dict.fromkeys(top_gene_symbols))[:15],  # dedup, preserve order
        is_cns_disease=is_cns,
        biological_context=_build_context(disease_name, top_gene_symbols, key_pathways),
        data_sources=data_sources,
    )

    logger.info(
        f"[Agent 1] Profile complete: {len(profile.known_targets)} targets, "
        f"{len(profile.key_pathways)} pathways, CNS={is_cns}"
    )
    return profile


def _infer_pathways(disease_name: str, gene_symbols: list[str]) -> list[str]:
    """Infer likely disease pathways from name and gene symbols."""
    from src.tools.kegg_client import get_disease_pathways_curated

    # Start with curated pathways for well-known diseases
    pathways = get_disease_pathways_curated(disease_name)

    # Add pathway hints based on gene symbols
    gene_pathway_hints = {
        "MTOR": "mTOR signaling pathway",
        "AMPK": "AMPK signaling pathway",
        "APP": "Amyloid precursor protein processing",
        "MAPT": "Tau protein phosphorylation",
        "APOE": "APOE-mediated lipid transport",
        "BACE1": "Beta-secretase cleavage",
        "PSEN1": "Presenilin/gamma-secretase complex",
        "PSEN2": "Presenilin/gamma-secretase complex",
        "TP53": "p53 tumor suppressor pathway",
        "KRAS": "RAS/MAPK signaling",
        "PIK3CA": "PI3K/AKT signaling",
        "EGFR": "EGFR receptor tyrosine kinase signaling",
        "NFKB1": "NF-kB inflammatory signaling",
        "TNF": "TNF/neuroinflammation",
        "IL6": "IL-6/JAK-STAT signaling",
        "HMGCR": "Cholesterol biosynthesis/mevalonate pathway",
        "PTGS2": "COX-2/prostaglandin synthesis",
    }

    extra_pathways = []
    for gene in gene_symbols[:10]:
        hint = gene_pathway_hints.get(gene.upper())
        if hint and hint not in pathways and hint not in extra_pathways:
            extra_pathways.append(hint)

    # Fall back to disease-name-based inference
    if not pathways and not extra_pathways:
        lower = disease_name.lower()
        if "alzheimer" in lower:
            extra_pathways = ["amyloid cascade", "tau phosphorylation", "neuroinflammation", "mTOR signaling"]
        elif "parkinson" in lower:
            extra_pathways = ["alpha-synuclein aggregation", "mitochondrial dysfunction", "UPS pathway"]
        elif "cancer" in lower or "carcinoma" in lower:
            extra_pathways = ["cell cycle regulation", "apoptosis", "angiogenesis", "MAPK signaling"]
        elif "diabetes" in lower:
            extra_pathways = ["insulin signaling", "glucose metabolism", "AMPK/mTOR axis"]

    all_pathways = [str(p) for p in (pathways + extra_pathways) if p]
    return list(dict.fromkeys(all_pathways))[:8]


def _build_context(disease: str, genes: list[str], pathways: list[str]) -> str:
    """Build a brief biological context summary."""
    gene_str = ", ".join(genes[:5]) if genes else "not yet characterized"
    path_str = "; ".join(pathways[:3]) if pathways else "unknown"
    return (
        f"{disease} is associated with key genes: {gene_str}. "
        f"Core pathogenic pathways include: {path_str}. "
        f"This information was derived from Open Targets and DisGeNET databases."
    )
