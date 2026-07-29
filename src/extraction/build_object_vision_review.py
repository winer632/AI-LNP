"""Build a visual, side-by-side review page for reconstructed objects."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .reconstruct_pdf_objects import OUTPUT
from .run_abstract_first import ROOT


def build() -> dict:
    manifest = json.loads((OUTPUT / "object_vision_manifest.json").read_text())
    cards: list[str] = []
    for item in manifest["objects"]:
        crop_src = (ROOT / item["crop_path"]).relative_to(OUTPUT)
        corrected_path = OUTPUT / "results" / f"{item['object_id']}.human_corrected.json"
        result_path = (
            corrected_path if corrected_path.exists()
            else OUTPUT / "results" / f"{item['object_id']}.validated.json"
        )
        result = json.loads(result_path.read_text())
        correction_note = (
            "<p class='corrected'><strong>Human-corrected:</strong> "
            "This display includes source-verified corrections. The original model "
            "result and correction ledger remain in the results folder.</p>"
            if corrected_path.exists() else ""
        )
        facts = "".join(
            "<tr>"
            f"<td>{html.escape(row['panel'] or '')}</td>"
            f"<td>{html.escape(row['population'])}</td>"
            f"<td>{html.escape(row['endpoint'])}</td>"
            f"<td>{html.escape(row['value'])} {html.escape(row['unit'] or '')}</td>"
            f"<td>{html.escape(row['support_kind'])}</td>"
            f"<td>{html.escape(row['visible_support'])}</td>"
            "</tr>"
            for row in result["printed_facts"]
        ) or "<tr><td colspan='6'>No printed numeric facts extracted.</td></tr>"
        raw_labels = "".join(
            "<tr>"
            f"<td>{html.escape(row['panel'])}</td>"
            f"<td>{html.escape(row['group'] or '')}</td>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(row['value'])} {html.escape(row['unit'] or '')}</td>"
            f"<td>{html.escape(row['label_type'])}</td>"
            "</tr>"
            for row in result["raw_panel_labels"]
        ) or "<tr><td colspan='5'>No raw labels transcribed.</td></tr>"
        comparisons = "".join(
            "<li>"
            f"<strong>{html.escape(row['subject_group'])}</strong> "
            f"{html.escape(row['direction'])} than/present versus "
            f"{html.escape(', '.join(row['comparator_groups']))}: "
            f"{html.escape(row['visible_support'])}"
            "</li>"
            for row in result["qualitative_comparisons"]
        ) or "<li>No qualitative comparison extracted.</li>"
        excluded = "".join(
            "<li>"
            f"{html.escape(row['panel'] or '')} {html.escape(row['endpoint'])}: "
            f"{html.escape(row['exclusion_reason'])}"
            "</li>"
            for row in result["excluded_estimates"]
        ) or "<li>No estimates excluded.</li>"
        cards.append(f"""
<section>
  <h2>{html.escape(item['object_id'])}</h2>
  <p><code>{html.escape(item['source_file'])}</code>, page {item['page']}, {html.escape(item['label'])}</p>
  <div class="grid">
    <div><img src="{html.escape(str(crop_src))}" alt="{html.escape(item['object_id'])} source crop"></div>
    <div>
      <p><strong>Readability:</strong> {html.escape(result['readability'])}</p>
      <p><strong>Status:</strong> {html.escape(item['status'])}</p>
      {correction_note}
      <h3>Raw visible labels</h3>
      <table><thead><tr><th>Panel</th><th>Group</th><th>Label</th><th>Value</th><th>Type</th></tr></thead><tbody>{raw_labels}</tbody></table>
      <h3>Printed facts</h3>
      <table><thead><tr><th>Panel</th><th>Population</th><th>Endpoint</th><th>Value</th><th>Support</th><th>Visible text</th></tr></thead><tbody>{facts}</tbody></table>
      <h3>Qualitative comparisons</h3><ul>{comparisons}</ul>
      <h3>Excluded estimates</h3><ul>{excluded}</ul>
    </div>
  </div>
</section>""")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Day 8 object-level vision review</title>
<style>
body{{font:15px system-ui;margin:24px;background:#f5f7f8;color:#172126}}
section{{background:white;padding:20px;margin:0 0 24px;border:1px solid #d9e0e3;border-radius:10px}}
.grid{{display:grid;grid-template-columns:minmax(380px,1fr) minmax(520px,1.4fr);gap:20px}}
img{{max-width:100%;border:1px solid #aab7bd}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd5d9;padding:6px;text-align:left;vertical-align:top}}
th{{background:#edf2f4}} code{{overflow-wrap:anywhere}} @media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
.corrected{{background:#fff4ce;border-left:4px solid #b7791f;padding:8px 10px}}
</style></head><body>
<h1>Day 8 object-level PDF review</h1>
<p>The left side is the exact reconstructed source crop; the right side separates printed facts, qualitative comparisons, and rejected estimates.</p>
{''.join(cards)}
</body></html>"""
    output = OUTPUT / "object_vision_review.html"
    output.write_text(page, encoding="utf-8")
    return {"review_path": str(output), "objects": len(cards)}



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
