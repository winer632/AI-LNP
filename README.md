# AI-LNP

AI-LNP is a literature-grounded system for finding and comparing lipid
nanoparticle (LNP) evidence for hepatocytes, Kupffer cells, liver sinusoidal
endothelial cells (LSECs), and hepatic stellate cells (HSCs).

The current engineering priority is reliable evidence recovery: every accepted
record must preserve its paper, experiment, biological context, outcome,
source location, and exact supporting evidence.

## Current status

The compact extraction workflow has been implemented and tested on nine
open-access gold-set papers.

| Measure | Current result |
|---|---:|
| Gold papers | 9 |
| Gold outcome records | 15 |
| Final recovered outcome records, one run | 11/15 (73.3%) |
| Final recovered outcome records, union plus vision | 12/15 (80.0%) |
| Final recovered outcome records, plus record-level salvage | 13/15 (86.7%) |
| Missing outcome records, one run | 4/15 (26.7%) |
| Local candidate-inventory recall | 15/15 |
| Current quality-gate status | **Not passed** |

Two recall figures, because they measure different things. A single run scores
the five result roots in `RESULT_ROOTS` and recovers 11/15 at precision 0.1897.
The union across extraction configurations, with the GP-004 and GP-006 panel
reads merged and `vision_relationship_polarity` on, recovers 12/15 at 0.2264.
Both are written by the same evaluator, so its report records `result_roots`
and says which it is. Quote the single-run figure unless you mean the ensemble.

The 13/15 figure is deterministic and rebuildable, not a side channel. It comes
from `codex_union_vision_v3`, which unions GP-004 against a record-level salvage
of the full-view runs and then merges two panel reads. Rebuild it with

```bash
AI_LNP_FLAG_RECORD_LEVEL_SALVAGE=1 \
  python -m src.extraction.build_union_vision_v3 --confirm-write
```

and score it with `vision_relationship_polarity` on. An earlier version of this
paragraph described 13/15 as a blind-adjudication side channel that could not be
reproduced byte for byte. That was true of a different 13/15, from an earlier
round; the current one reproduces exactly and is pinned by
`tests/test_union_vision_v3_rebuild.py`.

The 15/15 candidate-inventory result means local code can flag evidence groups
that may represent outcomes. It does **not** mean those outcomes have been
correctly extracted, validated, and merged. The single-run recovery result is 11/15, and 13/15 is reachable only by
combining configurations, so the compact route is not yet reliable enough for
unattended database expansion.

## Current compact workflow

```text
paper retrieval
  -> ingestion and provenance-preserving parsing
  -> lexical + semantic retrieval
  -> full local evidence inventory
  -> compact API packet
  -> local complexity assessment
  -> first structured LLM extraction
  -> ordinary schema/evidence validation
  -> complex-paper candidate and coverage check
  -> route unmatched or invalid evidence
       -> narrow text repair
       -> targeted table/figure vision
       -> human review when ambiguous
  -> deterministic merge
  -> validation and coverage recheck
  -> final result
  -> gold-set evaluation
```

### Main files

| Stage | Main file(s) | Purpose |
|---|---|---|
| Ingestion | `src/rag/ingestion.py`, `src/rag/run_pipeline.py` | Parse papers and retain source coordinates and provenance. |
| Compact packet | `src/rag/compact_api_packet.py` | Select high-value evidence while recording what the token budget excludes. |
| Structured packet | `src/rag/structured_compact_packet.py` | Spend the same budget differently: never drop a table, table row or caption, then fill the remainder by retrieval rank. Selected by the `structured_evidence_view` flag, which is on by default. |
| Complexity | `src/extraction/assess_outcome_complexity.py` | Classify a packet locally before extraction so complex papers receive coverage checking. |
| First extraction | `src/extraction/run_compact_one_call.py` | Make the first structured extraction call and save its candidate, validation, complexity, and coverage artifacts. |
| Ordinary validation | `src/extraction/compact_validation.py` | Check schema, identifiers, links, field states, and evidence references. |
| Outcome inventory | `build_full_outcome_inventory.py`, `build_outcome_candidates.py`, `consolidate_outcome_candidates.py` | Detect and deduplicate possible outcome groups using the larger local evidence view. |
| Coverage | `src/extraction/check_outcome_coverage.py` | Compare extracted records with candidate groups and keep unmatched groups explicit. |
| Routing | `src/extraction/route_compact_findings.py` | Separate field errors, missing text outcomes, missing visual outcomes, and ambiguous cases. |
| Text repair | `run_narrow_repair.py`, `run_missing_record_repair.py` | Repair one bounded field or recover one missing text-supported record. |
| Vision repair | `build_missing_record_vision_tasks.py`, `run_missing_record_vision.py` | Send only a targeted figure/table image and small context packet for visual recovery. |
| Merge | `merge_compact_results.py`, `merge_missing_records.py`, `merge_consolidated_gap_results.py`, `merge_structured_view_pass.py` | Merge validated additions without overwriting the original extraction. |
| Final evaluation | `src/extraction/evaluate_final_gold_dynamic.py` | Measure one-to-one recovery from the actual merged records. |

No validation or candidate-counting step automatically triggers a paid call.
Call-running commands require explicit confirmation and cache completed
responses to prevent accidental duplicate spending.

## Why the current result is not ideal

The compact packet reduced input cost, but it sometimes compressed a large
paper into too small or too text-oriented an evidence view. Schema validation
could prove that returned JSON was well formed, but it could not prove that the
model had returned every experiment in the paper. The later candidate and
coverage checks exposed silent omissions, yet detecting a missing group did not
itself reconstruct the exact table cells, figure labels, panel relationships,
or experiment boundaries needed for a final record.

Repeated recovery prompts could not reliably fix evidence that had never been
converted into a clear machine-readable form. This is why the next improvement
is structural document parsing and targeted local vision, rather than simply
sending more of the PDF to an LLM.

## Next extraction improvement: Docling + local VLM

The planned extraction route is:

```text
PDF / XML / supplements
  -> Docling document structure
       -> sections, captions, tables, cells, reading order
       -> figure and table object boundaries
  -> local vision-language model (Ollama + Gemma)
       -> printed labels and numeric values
       -> panel, legend, marker, and spatial relationships
       -> explicit uncertainty/abstention
  -> structured visual observations
  -> compact evidence packet containing only relevant observations
  -> LLM normalization into project contracts
  -> deterministic validation, coverage, and merge
```

Docling should recover layout and table structure. The local VLM should inspect
only the figures or table regions that require visual interpretation. Its
output should be structured observations—not unconstrained conclusions—and
must preserve object, panel, row, column, label, value, unit, and image
coordinates. Exact values are accepted only when visibly printed or
deterministically derived; estimates and ambiguity go to human review.

This approach can improve recall without sending every page to a paid model.
It must still be benchmarked against all 15 gold outcomes before it is trusted.

## Expanding literature discovery

The discovery stage should expand beyond PubMed and Europe PMC while retaining
deduplication and a complete search manifest.

```text
cell-specific query families
  -> PubMed + Europe PMC
  -> OpenAlex discovery and citation graph
  -> semantic similarity search
  -> backward references + forward citations
  -> DOI / PMID / PMCID / title normalization
  -> deduplication
  -> screening and source-priority ranking
```

- **OpenAlex** broadens discovery, supplies citation relationships, and can find
  relevant papers that use different terminology.
- **Semantic search** retrieves conceptually related work rather than relying
  only on exact keywords. Semantic Scholar may be evaluated subject to its
  current API terms and rate limits.
- **Backward and forward citation chaining** finds foundational and follow-up
  studies from already relevant papers.
- **LNP databases and atlases** are useful as formulation and nearest-neighbor
  leads. Their entries should not be treated as complete outcome evidence unless
  they link back to a verifiable primary paper.

Every source must preserve query text, filters, pagination, retrieval date,
identifiers, raw response provenance, and deduplication decisions.

## Full-text access

Use the following lawful priority order:

1. PMC, Europe PMC, publisher open access, and author manuscripts.
2. DOI and repository lookup, including institutional repositories.
3. Unpaywall-style open-access resolution.
4. University-library link resolver, proxy, or VPN using the user's authorized
   institutional account.
5. Interlibrary loan or an author copy when no accessible full text exists.
6. Abstract-only processing with an explicit full-text-unavailable status.

The system may help navigate a university library session that the user has
opened and authenticated, but it must not store credentials, bypass access
controls, or redistribute licensed PDFs. Licensed papers and derived evidence
must follow the institution's terms.

## Scientific boundaries

The application keeps these categories separate:

- **Reported evidence (`y`)**: a measurement explicitly supported by a paper.
- **Normalized/derived data**: a documented mechanical transformation.
- **Similarity result**: a related reported formulation, not a prediction.
- **Experimental suggestion (`X`)**: an untested candidate.
- **Model prediction (`y_hat`)**: a separately labelled estimate, never
  presented as reported evidence.

The application does not claim a universally best formulation, validated
four-cell prediction, prospective biological validation, or complete evidence
recovery while the gold-set gate remains below target.

## Local setup

Two chains, with different requirements. The evaluation chain needs nothing
beyond a clone; the ingestion chain needs network access twice.

**Evaluation — runs from a clone, no network, no services, no API quota:**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                       # 675 pass, 1 skips
.venv/bin/python -m src.extraction.evaluate_final_gold_dynamic --confirm-write
```

That reproduces the single-run figure. The three union figures are rebuilt by
`docs/extraction/union_provenance.md` and, for the 13/15 root:

```bash
AI_LNP_FLAG_RECORD_LEVEL_SALVAGE=1 \
  .venv/bin/python -m src.extraction.build_union_vision_v3 --confirm-write
```

**Ingestion — needs network, and a second environment:**

```bash
# 1. Fetch the open-access packages. They are gitignored: large, and their
#    licences differ from this repository's.
.venv/bin/python -m src.screening.retrieve_gold_oa_packages --confirm-write

# 2. Build the block corpus. --skip-missing-xml records a gap as a warning
#    instead of stopping after earlier papers are already written.
.venv/bin/python -m src.rag.ingestion --confirm-write --skip-missing-xml

# 3. Retrieval needs torch and faiss, which the extraction path does not.
python3 -m venv .venv-rag && .venv-rag/bin/pip install -r requirements-rag.txt
.venv-rag/bin/python -m src.rag.run_pipeline
```

Step 2 also reaches the UniParse service at `UNIPARSE_BASE_URL` when
`uniparse_ingestion` is on, which it is by default. If the service is
unreachable, ingestion degrades to the PyMuPDF path and records a warning
rather than failing.

Run local tests:

```bash
.venv/bin/python -m pytest -q
```

Local vision testing will use Ollama with a multimodal Gemma model. The model
download was intentionally stopped on July 28, 2026 and can be resumed with:

```bash
ollama pull gemma3:4b
```

API keys remain only in `.env`. The file is ignored by Git and must never be
committed.

## Current reference documents

- `docs/extraction/corrected_compact_workflow.md`
- `docs/extraction/outcome_complexity_workflow.md`
- `docs/extraction/union_provenance.md` — which run produced each of the union's
  51 records, and the command that rebuilds the union from a fresh clone
- `reports/extraction/final_gold_dynamic_v1/evaluation.json`
- `reports/extraction/day5_afternoon_g1/comparison.md`
