"""Test correct drug query"""
import requests, sys
sys.stdout.reconfigure(encoding='utf-8')

# Check Drug type fields too
q = '{ __type(name: "Drug") { fields { name } } }'
r = requests.post("https://api.platform.opentargets.org/api/v4/graphql", json={"query": q}, timeout=30)
drug_fields = [f["name"] for f in (r.json().get("data", {}).get("__type", {}).get("fields") or [])]
print("Drug fields:", drug_fields[:15])

# Now test the correct query
gql = """
{
  disease(efoId: "MONDO_0004975") {
    name
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          id
          name
          maximumClinicalTrialPhase
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
r2 = requests.post(
    "https://api.platform.opentargets.org/api/v4/graphql",
    json={"query": gql}, timeout=30
)
print(f"\nHTTP {r2.status_code}")
data = r2.json()
if "errors" in data:
    print("Errors:", [(e.get("message", "")[:120]) for e in data["errors"]])
else:
    disease = data.get("data", {}).get("disease", {})
    print(f"Disease: {disease.get('name')}")
    candidates = disease.get("drugAndClinicalCandidates", {})
    rows = candidates.get("rows", [])
    print(f"Total drugs: {candidates.get('count')}, returned: {len(rows)}")
    for row in rows[:10]:
        d = row.get("drug", {})
        moa_rows = d.get("mechanismsOfAction", {}).get("rows", [])
        moa = moa_rows[0].get("mechanismOfAction", "") if moa_rows else ""
        targets = [t.get("approvedSymbol") for r in moa_rows for t in r.get("targets", [])][:3]
        print(f"  {d.get('name')} | stage={row.get('maxClinicalStage')} | phase={d.get('maximumClinicalTrialPhase')} | {moa[:50]} | targets={targets}")
