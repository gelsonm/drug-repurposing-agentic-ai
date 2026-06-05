"""Patch all remaining int(float()) issues across the codebase"""
from pathlib import Path

fixes = [
    # kg_candidate.py: ChEMBL max_phase
    ("src/agents/kg_candidate.py",
     'max_phase = int(mol.get("max_phase", 0) or 0) if mol else 0',
     'max_phase = int(float(mol.get("max_phase", 0) or 0)) if mol else 0'),
]

for filepath, old, new in fixes:
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")
    if old in content:
        path.write_text(content.replace(old, new), encoding="utf-8")
        print(f"PATCHED: {filepath}")
    else:
        # Normalize CRLF and try again
        content_lf = content.replace("\r\n", "\n")
        old_lf = old.replace("\r\n", "\n")
        if old_lf in content_lf:
            path.write_text(content_lf.replace(old_lf, new), encoding="utf-8")
            print(f"PATCHED (LF): {filepath}")
        else:
            idx = content.find("max_phase = int(mol")
            print(f"NOT FOUND in {filepath}")
            print(repr(content[idx:idx+100]))
