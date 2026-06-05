"""Pydantic schemas for the drug repurposing v2 pipeline."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────

class EvidenceStrength(str, Enum):
    STRONG = "strong"       # ≥3 peer-reviewed sources
    MODERATE = "moderate"   # 1-2 peer-reviewed sources
    WEAK = "weak"           # Only computational / indirect
    UNKNOWN = "unknown"


class AlignmentType(str, Enum):
    DIRECT = "direct"         # Drug targets disease pathway gene directly
    INDIRECT = "indirect"     # Drug targets upstream/downstream regulator
    SPECULATIVE = "speculative"  # No known pathway connection found


class SafetyFlag(str, Enum):
    CLEAN = "clean"
    YELLOW = "yellow"         # Concerns but not disqualifying
    RED = "red"               # Disqualifying (failed trials / contraindicated)
    UNKNOWN = "unknown"


class ClinicalSignal(str, Enum):
    STRONG = "strong_evidence"
    EMERGING = "emerging_evidence"
    EARLY = "early_stage"
    MIXED = "mixed_evidence"
    NONE = "no_trials"


# ── Disease Profile ────────────────────────────────────────────────────────

class GeneAssociation(BaseModel):
    gene_symbol: str
    gene_id: str | None = None          # NCBI Gene ID
    association_score: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str = "Open Targets"
    evidence_count: int = 0


class DiseaseProfile(BaseModel):
    """Structured disease characterization produced by Agent 1."""
    disease_name: str
    efo_id: str | None = None           # EFO ontology ID (Open Targets)
    mondo_id: str | None = None         # MONDO disease ontology ID
    mesh_id: str | None = None

    key_pathways: list[str] = Field(default_factory=list)
    known_targets: list[GeneAssociation] = Field(default_factory=list)
    disease_genes: list[str] = Field(default_factory=list)  # top gene symbols
    biological_context: str | None = None

    # CNS flag — important for BBB check in pathway validator
    is_cns_disease: bool = False

    data_sources: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Candidate Drug ─────────────────────────────────────────────────────────

class KGEvidence(BaseModel):
    path_description: str
    path_score: float = Field(ge=0.0, le=1.0, default=0.0)
    shared_targets: list[str] = Field(default_factory=list)
    source: str = "Hetionet"


class CandidateDrug(BaseModel):
    """A drug candidate with initial KG evidence — output of Agent 2."""
    drug_name: str
    chembl_id: str | None = None
    drugbank_id: str | None = None      # DB##### format
    approval_status: str | None = None  # "approved", "investigational", etc.
    max_phase: int | None = None        # ChEMBL max clinical phase
    original_indication: str | None = None
    mechanism_of_action: str | None = None
    known_targets: list[str] = Field(default_factory=list)  # gene symbols

    # KG evidence
    kg_evidence: list[KGEvidence] = Field(default_factory=list)
    network_proximity_score: float = Field(ge=0.0, le=1.0, default=0.0)
    open_targets_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Evidence from later agents (populated progressively)
    literature_pmids: list[str] = Field(default_factory=list)
    literature_positive_count: int = 0
    literature_negative_count: int = 0

    # Scores (filled by scorer agent)
    moa_alignment_score: float | None = None
    pathway_plausibility_score: float | None = None
    adversarial_adjustment: float = 0.0  # negative adjustment


# ── Pathway Alignment ──────────────────────────────────────────────────────

class PathwayAlignment(BaseModel):
    """Output of Agent 5 (Pathway Validator) for a single drug-disease pair."""
    drug_name: str
    disease_name: str

    alignment_type: AlignmentType = AlignmentType.SPECULATIVE
    connection_description: str | None = None

    # Pathway IDs
    kegg_pathway_ids: list[str] = Field(default_factory=list)
    reactome_pathway_ids: list[str] = Field(default_factory=list)

    biological_plausibility_score: float = Field(ge=0.0, le=1.0, default=0.0)
    validation_notes: str | None = None

    # BBB / tissue access check
    bbb_penetrant: bool | None = None  # None = unknown
    tissue_distribution_note: str | None = None

    evidence_pmids: list[str] = Field(default_factory=list)


# ── Adversarial Report ─────────────────────────────────────────────────────

class FailedTrial(BaseModel):
    nct_id: str
    title: str | None = None
    phase: str | None = None
    why_stopped: str | None = None


class AdversarialReport(BaseModel):
    """Output of Agent 6 (Adversarial Critic) for a single candidate."""
    drug_name: str
    disease_name: str

    red_flags: list[str] = Field(default_factory=list)
    yellow_flags: list[str] = Field(default_factory=list)
    failed_trials: list[FailedTrial] = Field(default_factory=list)

    overall_verdict: str = "UNKNOWN"
    confidence_adjustment: float = Field(ge=-1.0, le=0.0, default=0.0)
    safety_flag: SafetyFlag = SafetyFlag.UNKNOWN


# ── Scored Candidate (final) ───────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    kg_proximity: float = Field(ge=0.0, le=1.0, default=0.0)
    moa_alignment: float = Field(ge=0.0, le=1.0, default=0.0)
    literature: float = Field(ge=0.0, le=1.0, default=0.0)
    pathway_plausibility: float = Field(ge=0.0, le=1.0, default=0.0)
    adversarial_adjustment: float = Field(ge=-1.0, le=0.0, default=0.0)
    composite: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_interval: float = Field(ge=0.0, default=0.1)  # ±CI

    @property
    def rank_label(self) -> str:
        s = self.composite
        if s >= 0.75: return "Strong"
        if s >= 0.55: return "Moderate"
        if s >= 0.35: return "Weak"
        return "Speculative"


class ScoredCandidate(BaseModel):
    """A fully evaluated, scored repurposing candidate."""
    candidate: CandidateDrug
    pathway_alignment: PathwayAlignment | None = None
    adversarial_report: AdversarialReport | None = None
    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    one_line_justification: str | None = None
    recommended_next_step: str | None = None
    all_source_ids: list[str] = Field(default_factory=list)  # DB IDs + PMIDs


# ── Final Report ───────────────────────────────────────────────────────────

class RepurposingReport(BaseModel):
    """Complete output of the 8-agent pipeline."""
    disease_name: str
    disease_profile: DiseaseProfile | None = None
    candidates: list[ScoredCandidate] = Field(default_factory=list)

    executive_summary: str | None = None
    methodology_notes: str | None = None
    limitations: list[str] = Field(default_factory=list)
    all_references: list[str] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str = "2.0.0"
    total_runtime_seconds: float | None = None

    @property
    def top_candidates(self) -> list[ScoredCandidate]:
        return sorted(self.candidates, key=lambda c: c.score.composite, reverse=True)
