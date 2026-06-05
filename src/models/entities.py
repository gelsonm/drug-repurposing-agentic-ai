"""Pydantic data models for drug repurposing entities."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    MECHANISTIC = "mechanistic"
    LITERATURE = "literature"
    CLINICAL_TRIAL = "clinical_trial"
    SAFETY_SIGNAL = "safety_signal"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class ClinicalFeasibility(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class SafetyFlag(str, Enum):
    CLEAN = "clean"
    MINOR_CONCERNS = "minor_concerns"
    MAJOR_CONCERNS = "major_concerns"
    CONTRAINDICATED = "contraindicated"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Core Entities
# ---------------------------------------------------------------------------

class Drug(BaseModel):
    """Represents an approved or investigational drug."""

    name: str
    chembl_id: str | None = None
    drugbank_id: str | None = None
    pubchem_cid: int | None = None
    smiles: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    mechanism_of_action: str | None = None
    targets: list[str] = Field(default_factory=list)  # UniProt IDs
    approved_indications: list[str] = Field(default_factory=list)
    drug_class: str | None = None
    max_phase: int | None = None  # ChEMBL max clinical phase


class Disease(BaseModel):
    """Represents a disease or condition."""

    name: str
    mondo_id: str | None = None  # MONDO Disease Ontology ID
    mesh_id: str | None = None
    omim_id: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    description: str | None = None
    gene_associations: list[str] = Field(default_factory=list)  # gene symbols
    is_rare: bool = False


class EvidenceItem(BaseModel):
    """A single piece of evidence supporting a drug-disease association."""

    evidence_type: EvidenceType
    source: str  # e.g. "PubMed", "ChEMBL", "Hetionet"
    source_id: str | None = None  # e.g. PMID, ChEMBL assay ID
    title: str | None = None
    snippet: str  # relevant text passage
    url: str | None = None
    year: int | None = None
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)


class KGPath(BaseModel):
    """A reasoning path through the knowledge graph."""

    nodes: list[str]  # node names
    edge_types: list[str]  # relationship types
    path_score: float = Field(ge=0.0, le=1.0, default=0.0)
    description: str | None = None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    """Detailed breakdown of composite repurposing score."""

    mechanistic_score: float = Field(ge=0.0, le=1.0, default=0.0)
    literature_score: float = Field(ge=0.0, le=1.0, default=0.0)
    clinical_feasibility_score: float = Field(ge=0.0, le=1.0, default=0.0)
    data_confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    composite_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Metadata
    kg_paths_found: int = 0
    pubmed_hits: int = 0
    active_trials: int = 0
    safety_flag: SafetyFlag = SafetyFlag.UNKNOWN
    clinical_feasibility: ClinicalFeasibility = ClinicalFeasibility.UNKNOWN

    def compute_composite(
        self,
        w_mech: float = 0.35,
        w_lit: float = 0.30,
        w_clin: float = 0.20,
        w_conf: float = 0.15,
    ) -> float:
        """Compute and store the composite score."""
        self.composite_score = (
            w_mech * self.mechanistic_score
            + w_lit * self.literature_score
            + w_clin * self.clinical_feasibility_score
            + w_conf * self.data_confidence_score
        )
        return self.composite_score


# ---------------------------------------------------------------------------
# Main Output Entity
# ---------------------------------------------------------------------------

class RepurposingCandidate(BaseModel):
    """A ranked drug-disease repurposing candidate with full evidence chain."""

    drug: Drug
    disease: Disease
    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)

    # Evidence
    kg_paths: list[KGPath] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)

    # Analysis outputs
    mechanistic_rationale: str | None = None
    literature_summary: str | None = None
    clinical_summary: str | None = None
    safety_summary: str | None = None
    overall_assessment: str | None = None

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str = "1.0.0"

    @property
    def rank_label(self) -> str:
        """Human-readable strength label based on composite score."""
        s = self.score.composite_score
        if s >= 0.75:
            return "Strong"
        elif s >= 0.55:
            return "Moderate"
        elif s >= 0.35:
            return "Weak"
        return "Speculative"

    @property
    def top_citations(self) -> list[EvidenceItem]:
        """Return top 5 evidence items by relevance."""
        return sorted(
            self.evidence_items, key=lambda e: e.relevance_score, reverse=True
        )[:5]


class RepurposingReport(BaseModel):
    """Full report output from the pipeline."""

    query_drug: str | None = None
    query_disease: str | None = None
    query_raw: str

    candidates: list[RepurposingCandidate] = Field(default_factory=list)
    summary: str | None = None
    next_steps: list[str] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str = "1.0.0"
    total_runtime_seconds: float | None = None

    @property
    def top_candidates(self) -> list[RepurposingCandidate]:
        """Return candidates sorted by composite score."""
        return sorted(
            self.candidates,
            key=lambda c: c.score.composite_score,
            reverse=True,
        )
