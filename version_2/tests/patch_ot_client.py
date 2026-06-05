"""Patch max_phase conversion in open_targets_client.py"""
from pathlib import Path

path = Path("src/tools/open_targets_client.py")
content = path.read_text(encoding="utf-8")

# Fix the int() conversion that crashes on 'UNKNOWN'
old = """        max_phase = int(
            drug.get("maximumClinicalStage") or row.get("maxClinicalStage") or 0
        )"""

new = """        raw_phase = drug.get("maximumClinicalStage") or row.get("maxClinicalStage") or 0
        try:
            max_phase = int(float(raw_phase))
        except (ValueError, TypeError):
            max_phase = 0"""

if old in content:
    content = content.replace(old, new)
    path.write_text(content, encoding="utf-8")
    print("PATCHED max_phase OK")
else:
    print("Pattern not found")
    # Find and show context
    idx = content.find("max_phase = int(")
    print(repr(content[idx:idx+150]))
