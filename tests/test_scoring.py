"""
Tests for the composite scoring engine.
These are pure unit tests — no API calls, no LLM, no external services.
"""
import pytest
from src.scoring.composite_scorer import (
    compute_composite_score,
    score_clinical,
    score_data_confidence,
    score_literature,
    score_mechanistic,
)


class TestMechanisticScoring:
    def test_no_evidence(self):
        score = score_mechanistic([], 0.0, 0)
        assert score == 0.0

    def test_strong_paths(self):
        paths = [
            {"path_score": 0.9, "description": "Drug→Gene→Disease"},
            {"path_score": 0.7, "description": "Drug→Gene→Pathway→Disease"},
        ]
        score = score_mechanistic(paths, gene_overlap=0.5, n_targets=5)
        assert score > 0.5

    def test_gene_overlap_boosts_score(self):
        base = score_mechanistic([], 0.0, 5)
        boosted = score_mechanistic([], 0.8, 5)
        assert boosted > base

    def test_score_range(self):
        paths = [{"path_score": 1.0}] * 10
        score = score_mechanistic(paths, 1.0, 50)
        assert 0.0 <= score <= 1.0


class TestLiteratureScoring:
    def test_empty_evidence(self):
        score = score_literature([], 0)
        assert score == 0.0

    def test_recent_articles_bonus(self):
        passages_old = [{"relevance_score": 0.7, "year": "2010"}]
        passages_new = [{"relevance_score": 0.7, "year": "2024"}]
        score_old = score_literature(passages_old, 5)
        score_new = score_literature(passages_new, 5)
        assert score_new > score_old

    def test_pubmed_count_matters(self):
        score_few = score_literature([], 2)
        score_many = score_literature([], 50)
        assert score_many > score_few

    def test_score_range(self):
        passages = [{"relevance_score": 0.9, "year": "2024"}] * 10
        score = score_literature(passages, 100)
        assert 0.0 <= score <= 1.0


class TestClinicalScoring:
    def test_no_trials(self):
        landscape = {"clinical_signal": "no_trials", "total": 0, "active": 0, "completed": 0, "terminated": 0}
        score, feasibility = score_clinical(landscape, "clean")
        assert score < 0.5

    def test_strong_evidence(self):
        landscape = {"clinical_signal": "strong_evidence", "total": 5, "active": 2, "completed": 3, "terminated": 0}
        score, _ = score_clinical(landscape, "clean")
        assert score >= 0.7

    def test_contraindicated_kills_score(self):
        landscape = {"clinical_signal": "strong_evidence", "total": 3, "active": 1, "completed": 2, "terminated": 0}
        score, _ = score_clinical(landscape, "contraindicated")
        assert score == 0.0

    def test_high_termination_penalty(self):
        landscape_ok = {"clinical_signal": "emerging_evidence", "total": 4, "active": 2, "completed": 2, "terminated": 0}
        landscape_bad = {"clinical_signal": "emerging_evidence", "total": 4, "active": 0, "completed": 1, "terminated": 3}
        score_ok, _ = score_clinical(landscape_ok, "clean")
        score_bad, _ = score_clinical(landscape_bad, "clean")
        assert score_ok > score_bad


class TestDataConfidence:
    def test_all_missing(self):
        score = score_data_confidence(False, False, False, False, 0, 0)
        assert score == 0.0

    def test_all_present(self):
        score = score_data_confidence(True, True, True, True, 20, 5)
        assert score == 1.0

    def test_partial(self):
        score = score_data_confidence(True, True, False, False, 3, 2)
        assert 0.0 < score < 1.0


class TestCompositeScorer:
    def test_full_pipeline(self):
        landscape = {
            "clinical_signal": "emerging_evidence",
            "total": 5, "active": 2, "completed": 2, "terminated": 1,
        }
        breakdown = compute_composite_score(
            kg_paths=[{"path_score": 0.8, "description": "test path"}],
            gene_overlap=0.3,
            n_targets=8,
            rag_passages=[{"relevance_score": 0.75, "year": "2023", "text": "test"}],
            pubmed_count=12,
            trial_landscape=landscape,
            safety_flag="clean",
            has_chembl=True,
            has_uniprot=True,
            has_label=True,
        )
        assert 0.0 <= breakdown.composite_score <= 1.0
        assert breakdown.mechanistic_score > 0
        assert breakdown.literature_score > 0

    def test_weights_sum_to_one(self):
        """Verify the default weights (0.35+0.30+0.20+0.15) sum to 1.0"""
        from src.config import get_settings
        s = get_settings()
        total = s.weight_mechanistic + s.weight_literature + s.weight_clinical + s.weight_data_confidence
        assert abs(total - 1.0) < 0.001
