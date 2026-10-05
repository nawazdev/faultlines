"""Build notebooks/faultlines_official_addon.py: one paste-in cell for a copy of the official starter.

The FaultLines package is embedded as a base64 zip so the cell needs no internet and no extra inputs.
"""

import base64
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted((ROOT / "src").rglob("*")):
        rel = p.relative_to(ROOT / "src" / "faultlines") if p.is_relative_to(ROOT / "src" / "faultlines") else p
        needed = bool(rel.parts) and rel.parts[0] in ("atlas", "toolguard", "__init__.py")   # no mini-harness needed
        if needed and p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in str(p):
            z.write(p, p.relative_to(ROOT))
payload = base64.b64encode(buf.getvalue()).decode()
tpl = (ROOT / "scripts" / "official_addon_template.py").read_text()
assert tpl.count('"__PAYLOAD__"') == 1
out = ROOT / "notebooks" / "faultlines_official_addon.py"
out.write_text(tpl.replace('"__PAYLOAD__"', f'"{payload}"'))
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
