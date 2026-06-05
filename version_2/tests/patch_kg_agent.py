"""Patch OT client: fix phase string parsing; patch KG agent: fix phase filter and lower threshold"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# ── Fix 1: OT client phase parsing ────────────────────────────────────────
ot_path = Path("src/tools/open_targets_client.py")
content = ot_path.read_text(encoding="utf-8")

# Replace the raw_phase / int conversion
old_phase = """        raw_phase = drug.get("maximumClinicalStage") or row.get("maxClinicalStage") or 0
        try:
            max_phase = int(float(raw_phase))
        except (ValueError, TypeError):
            max_phase = 0"""

new_phase = """        # OT v4 returns string stages like "APPROVAL", "PHASE_3", "PHASE_2_3"
        raw_phase = drug.get("maximumClinicalStage") or row.get("maxClinicalStage") or ""
        max_phase = _parse_clinical_stage(raw_phase)"""

if old_phase in content:
    # Add helper function before get_known_drugs_for_disease
    helper = '''
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
    nums = re.findall(r"\\d+", s)
    return int(nums[-1]) if nums else 0

'''
    # Insert helper before the function
    insert_before = "def get_known_drugs_for_disease"
    content = content.replace(insert_before, helper + insert_before)
    content = content.replace(old_phase, new_phase)
    ot_path.write_text(content, encoding="utf-8")
    print("PATCHED: OT phase parsing OK")
else:
    print("Phase pattern not found — searching:")
    idx = content.find("raw_phase")
    print(repr(content[idx:idx+200]))

# ── Fix 2: KG agent: lower phase filter from >=3 to >=2 ──────────────────
kg_path = Path("src/agents/kg_candidate.py")
content_kg = kg_path.read_text(encoding="utf-8")

old_filter = "if not name or phase < 3:  # Only Phase III+ approved"
new_filter = "if not name or phase < 2:  # Phase II and above (includes approved)"

if old_filter in content_kg:
    content_kg = content_kg.replace(old_filter, new_filter)
    kg_path.write_text(content_kg, encoding="utf-8")
    print("PATCHED: KG phase filter lowered to >=2 OK")
else:
    print("Phase filter pattern not found in kg_candidate.py")
    idx = content_kg.find("phase < ")
    print(repr(content_kg[idx:idx+80]))

print()
print("Done. Now test parsing:")
# Quick test
def _parse_clinical_stage(stage):
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
    import re
    nums = re.findall(r"\d+", s)
    return int(nums[-1]) if nums else 0

import re
for s in ["APPROVAL", "PHASE_3", "PHASE_2_3", "PHASE_2", "PHASE_1_2", "UNKNOWN", "", "PHASE_3_4"]:
    print(f"  {s!r} -> {_parse_clinical_stage(s)}")
