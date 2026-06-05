"""Debug script - run from project root"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

# 1. Check what's in the sample KG
import json
from pathlib import Path

kg_path = Path("data/kg/sample_kg.json")
if kg_path.exists():
    with open(kg_path) as f:
        kg = json.load(f)
    print(f"KG file: {len(kg.get('nodes',[]))} nodes, {len(kg.get('edges',[]))} edges")
    diseases = [n for n in kg.get('nodes',[]) if n.get('kind') == 'Disease']
    compounds = [n for n in kg.get('nodes',[]) if n.get('kind') == 'Compound']
    print(f"Disease nodes: {[d.get('name') for d in diseases[:10]]}")
    print(f"Compound nodes: {[c.get('name') for c in compounds[:10]]}")
else:
    print("No KG file found — using demo graph")

print()

# 2. Test OT knownDrugs query
print("Testing Open Targets knownDrugs with MONDO_0004975 ...")
from src.tools.http_utils import get_json, APIError
import requests

gql = """
query KnownDrugs($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    knownDrugs(size: $size) {
      count
      rows {
        drug {
          id
          name
          maximumClinicalTrialPhase
        }
        phase
        status
      }
    }
  }
}
"""
try:
    r = requests.post(
        "https://api.platform.opentargets.org/api/v4/graphql",
        json={"query": gql, "variables": {"efoId": "MONDO_0004975", "size": 10}},
        timeout=30
    )
    print(f"HTTP status: {r.status_code}")
    if r.status_code == 400:
        print(f"Error: {r.text[:500]}")
    else:
        data = r.json()
        rows = data.get("data", {}).get("disease", {}).get("knownDrugs", {}).get("rows", [])
        count = data.get("data", {}).get("disease", {}).get("knownDrugs", {}).get("count", 0)
        print(f"knownDrugs count: {count}, rows returned: {len(rows)}")
        for row in rows[:5]:
            d = row.get("drug", {})
            print(f"  - {d.get('name')} (phase {row.get('phase')}, status {row.get('status')})")
except Exception as e:
    print(f"Error: {e}")

print()

# 3. Test with EFO ID instead
print("Testing with EFO_0000249 (Alzheimers EFO) ...")
try:
    r2 = requests.post(
        "https://api.platform.opentargets.org/api/v4/graphql",
        json={"query": gql, "variables": {"efoId": "EFO_0000249", "size": 10}},
        timeout=30
    )
    print(f"HTTP status: {r2.status_code}")
    if r2.status_code == 400:
        print(f"Error: {r2.text[:500]}")
    else:
        data2 = r2.json()
        rows2 = data2.get("data", {}).get("disease", {}).get("knownDrugs", {}).get("rows", [])
        count2 = data2.get("data", {}).get("disease", {}).get("knownDrugs", {}).get("count", 0)
        print(f"knownDrugs count: {count2}, rows returned: {len(rows2)}")
        for row in rows2[:5]:
            d = row.get("drug", {})
            print(f"  - {d.get('name')} (phase {row.get('phase')}, status {row.get('status')})")
except Exception as e:
    print(f"Error: {e}")
