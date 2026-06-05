"""Tools package — all framework-agnostic API clients."""
from src.tools.chembl_client import (
    get_drug_by_chembl_id,
    get_drug_mechanisms,
    get_drug_targets,
    search_drug_by_name,
    extract_uniprot_ids,
)
from src.tools.clinical_trials_client import (
    TrialStatus,
    analyze_trial_landscape,
    get_trial_by_nct_id,
    search_trials,
)
from src.tools.http_utils import APIError, get_json, get_text
from src.tools.kg_client import (
    compute_gene_overlap_score,
    find_drug_disease_paths,
    get_disease_genes,
    get_drug_targets_from_kg,
    load_knowledge_graph,
)
from src.tools.openfda_client import (
    assess_safety_flag,
    extract_safety_summary,
    get_drug_label,
    get_top_adverse_reactions,
    search_adverse_events,
)
from src.tools.pubmed_client import (
    build_repurposing_query,
    fetch_abstracts,
    search_and_fetch,
    search_pubmed,
)
from src.tools.rag_client import (
    clear_collection,
    get_collection_stats,
    ingest_articles,
    query_literature,
)
from src.tools.uniprot_client import (
    extract_disease_associations,
    extract_gene_names,
    extract_protein_function,
    get_protein_by_accession,
    get_proteins_for_targets,
    search_proteins,
)

__all__ = [
    # ChEMBL
    "get_drug_by_chembl_id", "get_drug_mechanisms", "get_drug_targets",
    "search_drug_by_name", "extract_uniprot_ids",
    # ClinicalTrials
    "TrialStatus", "analyze_trial_landscape", "get_trial_by_nct_id", "search_trials",
    # HTTP
    "APIError", "get_json", "get_text",
    # KG
    "compute_gene_overlap_score", "find_drug_disease_paths", "get_disease_genes",
    "get_drug_targets_from_kg", "load_knowledge_graph",
    # OpenFDA
    "assess_safety_flag", "extract_safety_summary", "get_drug_label",
    "get_top_adverse_reactions", "search_adverse_events",
    # PubMed
    "build_repurposing_query", "fetch_abstracts", "search_and_fetch", "search_pubmed",
    # RAG
    "clear_collection", "get_collection_stats", "ingest_articles", "query_literature",
    # UniProt
    "extract_disease_associations", "extract_gene_names", "extract_protein_function",
    "get_protein_by_accession", "get_proteins_for_targets", "search_proteins",
]
