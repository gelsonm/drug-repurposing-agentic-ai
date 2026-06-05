"""Tests for the knowledge graph client."""
import pytest
from src.tools.kg_client import (
    _create_demo_graph,
    compute_gene_overlap_score,
    find_drug_disease_paths,
    get_disease_genes,
    get_drug_targets_from_kg,
    load_knowledge_graph,
)


class TestDemoGraph:
    def test_demo_graph_loads(self):
        G = _create_demo_graph()
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_demo_graph_has_metformin(self):
        G = _create_demo_graph()
        node_names = [d.get("name", "") for _, d in G.nodes(data=True)]
        assert "Metformin" in node_names

    def test_demo_graph_has_alzheimers(self):
        G = _create_demo_graph()
        node_names = [d.get("name", "") for _, d in G.nodes(data=True)]
        assert "Alzheimer's disease" in node_names


class TestKGQueries:
    def setup_method(self):
        """Use demo graph for all tests."""
        import src.tools.kg_client as kg_module
        kg_module._graph = _create_demo_graph()

    def test_find_drug_targets(self):
        targets = get_drug_targets_from_kg("Metformin")
        assert len(targets) > 0
        gene_names = [t["gene_name"] for t in targets]
        assert "AMPK" in gene_names

    def test_find_disease_genes(self):
        genes = get_disease_genes("Alzheimer's disease")
        assert len(genes) > 0
        gene_names = [g["gene_name"] for g in genes]
        assert any(g in gene_names for g in ["AMPK", "MTOR", "BACE1", "APP"])

    def test_find_kg_paths(self):
        paths = find_drug_disease_paths("Metformin", "Alzheimer's disease")
        assert len(paths) > 0
        # All paths should have nodes and edge_types
        for path in paths:
            assert "nodes" in path
            assert "edge_types" in path
            assert "path_score" in path

    def test_gene_overlap_score(self):
        score = compute_gene_overlap_score("Metformin", "Alzheimer's disease")
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should find some overlap

    def test_gene_overlap_unknown_drug(self):
        score = compute_gene_overlap_score("UnknownDrugXYZ", "Alzheimer's disease")
        assert score == 0.0

    def test_path_score_range(self):
        paths = find_drug_disease_paths("Metformin", "Alzheimer's disease")
        for path in paths:
            assert 0.0 <= path["path_score"] <= 1.0
