"""
Open Targets Platform GraphQL API client.
Documentation: https://platform.opentargets.org/api
No authentication required. Rate limit: generous (public API).

Used by:
  - Agent 1 (Disease Intel): disease characterization, known targets
  - Agent 2 (KG Candidate): drug-disease association scores
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.tools.http_utils import APIError, post_json

_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def _query(gql: str, variables: dict | None = None) -> dict[str, Any]:
    """Execute a GraphQL query against Open Targets Platform."""
    payload = {"query": gql, "variables": variables or {}}
    try:
        result = post_json(_GRAPHQL_URL, json=payload)
        if "errors" in result:
            logger.warning(f"Open Targets GraphQL errors: {result['errors']}")
        return result.get("data", {})
    except APIError as e:
        logger.warning(f"Open Targets query failed: {e}")
        return {}


def get_disease_info(disease_name: str) -> dict[str, Any]:
    """
    Search for a disease by name and return its EFO ID + basic metadata.

    Returns dict with: efoId, name, description, therapeuticAreas
    """
    gql = """
    query DiseaseSearch($q: String!) {
      search(queryString: $q, entityNames: ["disease"]) {
        hits {
          id
          name
          entity
          object {
            ... on Disease {
              id
              name
              description
              therapeuticAreas { id name }
            }
          }
        }
      }
    }
    """
    data = _query(gql, {"q": disease_name})
    hits = data.get("search", {}).get("hits", [])
    if hits:
        obj = hits[0].get("object", {})
        logger.info(f"Open Targets: found disease '{obj.get('name')}' (EFO: {obj.get('id')})")
        return obj
    logger.warning(f"Open Targets: disease '{disease_name}' not found")
    return {}


def get_disease_targets(efo_id: str, max_results: int = 20) -> list[dict[str, Any]]:
    """
    Get top gene targets associated with a disease from Open Targets.

    Returns list of {targetId, symbol, score, approvedName}.
    """
    gql = """
    query DiseaseTargets($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: { index: 0, size: $size }) {
          rows {
            target {
              id
              approvedSymbol
              approvedName
            }
            score
          }
        }
      }
    }
    """
    data = _query(gql, {"efoId": efo_id, "size": max_results})
    rows = data.get("disease", {}).get("associatedTargets", {}).get("rows", [])

    targets = []
    for row in rows:
        tgt = row.get("target", {})
        targets.append({
            "target_id": tgt.get("id"),
            "symbol": tgt.get("approvedSymbol"),
            "approved_name": tgt.get("approvedName"),
            "association_score": row.get("score", 0.0),
        })

    logger.info(f"Open Targets: {len(targets)} targets for EFO {efo_id}")
    return targets



def _parse_clinical_stage(stage: str) -> int:
    """Convert OT clinical stage string to integer phase number."""
    if not stage:
        return 0
    s = str(stage).upper().replace(" ", "_")
    if s in ("APPROVAL", "APPROVED"):
        return 4
    if s.startswith("PHASE_3") or s == "PHASE_2_3":
        return 3
    if s.startswith("PHASE_2"):
        return 2
    if s.startswith("PHASE_1") or s == "PHASE_1_2":
        return 1
    # Try numeric fallback
    import re
    nums = re.findall(r"\d+", s)
    return int(nums[-1]) if nums else 0

def get_known_drugs_for_disease(efo_id: str, max_results: int = 50) -> list[dict[str, Any]]:
    """
    Get drug candidates associated with a disease via Open Targets v4 API.
    Uses drugAndClinicalCandidates (replaces deprecated knownDrugs field).

    Returns list of {drug_name, chembl_id, max_phase, mechanism_of_action, target_symbols}.
    """
    gql = """
    query DrugCandidates($efoId: String!) {
      disease(efoId: $efoId) {
        drugAndClinicalCandidates {
          count
          rows {
            id
            maxClinicalStage
            drug {
              id
              name
              maximumClinicalStage
              mechanismsOfAction {
                rows {
                  mechanismOfAction
                  actionType
                  targets { approvedSymbol }
                }
              }
            }
          }
        }
      }
    }
    """
    data = _query(gql, {"efoId": efo_id})
    candidates = (data.get("disease") or {}).get("drugAndClinicalCandidates") or {}
    rows = candidates.get("rows") or []

    drugs = []
    seen: set = set()
    for row in rows:
        drug = row.get("drug") or {}
        name = drug.get("name", "")
        if not name or name in seen:
            continue
        seen.add(name)

        # OT v4 returns string stages like "APPROVAL", "PHASE_3", "PHASE_2_3"
        raw_phase = drug.get("maximumClinicalStage") or row.get("maxClinicalStage") or ""
        max_phase = _parse_clinical_stage(raw_phase)

        moa_rows = (drug.get("mechanismsOfAction") or {}).get("rows") or []
        moa_text = "; ".join(
            r.get("mechanismOfAction", "") for r in moa_rows[:2]
            if r.get("mechanismOfAction")
        )
        targets = list({
            t.get("approvedSymbol", "")
            for mrow in moa_rows
            for t in (mrow.get("targets") or [])
            if t.get("approvedSymbol")
        })

        drugs.append({
            "drug_name": name,
            "chembl_id": drug.get("id"),
            "max_phase": max_phase,
            "mechanism_of_action": moa_text or None,
            "target_symbols": targets,
        })

    logger.info(
        f"Open Targets: {len(drugs)} drug candidates for {efo_id} "
        f"(API total: {candidates.get('count', '?')})"
    )
    return drugs


def get_target_pathways(target_id: str) -> list[dict[str, Any]]:
    """
    Get pathway memberships for a target (Reactome pathways).

    Returns list of {pathway_id, pathway_name}.
    """
    gql = """
    query TargetPathways($ensgId: String!) {
      target(ensemblId: $ensgId) {
        id
        pathways {
          pathway
          pathwayId
        }
      }
    }
    """
    data = _query(gql, {"ensgId": target_id})
    pathways = data.get("target", {}).get("pathways", [])
    return [{"pathway_id": p.get("pathwayId"), "pathway_name": p.get("pathway")} for p in pathways]


def get_drug_disease_score(chembl_id: str, efo_id: str) -> float:
    """
    Get the Open Targets association score between a specific drug and disease.
    Returns float in [0, 1], or 0.0 if no association found.
    """
    gql = """
    query DrugDiseaseScore($chemblId: String!) {
      drug(chemblId: $chemblId) {
        id
        name
        indications {
          rows {
            disease { id name }
            maxPhaseForIndication
          }
        }
      }
    }
    """
    data = _query(gql, {"chemblId": chembl_id})
    rows = data.get("drug", {}).get("indications", {}).get("rows", [])
    for row in rows:
        dis = row.get("disease", {})
        if efo_id and dis.get("id") == efo_id:
            phase = row.get("maxPhaseForIndication", 0) or 0
            return min(1.0, phase / 4.0)
    return 0.0
