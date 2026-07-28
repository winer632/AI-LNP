# Corrected compact extraction workflow

This document separates three questions that must never be confused:

1. Was the returned JSON structurally valid?
2. Did the local evidence inventory notice every plausible outcome group?
3. Was every credible group actually recovered into a validated final record?

## End-to-end path

| Stage | Main file | What it does | Paid call? |
|---|---|---|---|
| Ingest | `src/ingestion.py` / pipeline ingestion modules | Parse the paper and preserve text, tables, figures, locations, and identifiers. | No |
| Compact evidence view | `src/rag/compact_api_packet.py` | Select a token-budgeted slice of retrieval evidence for the first extraction call. | No |
| Full local evidence view | `src/extraction/build_full_outcome_inventory.py` | Rejoins all locally parsed blocks so neither extraction nor recall checking is limited by the first-call token budget. | No |
| Full API packet | `src/rag/full_api_packet.py` | Writes that full view to `data/staging/rag/full_api_packets_v1/{paper_id}.json` with a manifest, in the same packet type and checksum rule as the compact packet. | No |
| Complexity census | `src/extraction/assess_outcome_complexity.py` | Before the first call, label the paper simple or complex from inexpensive signals, using thresholds calibrated per evidence view. | No |
| First extraction | `src/extraction/run_compact_one_call.py` | Extract records from the selected evidence view: `--evidence-view compact` for the budgeted packet, `--evidence-view full` for the complete local evidence. | Yes, one call per uncached paper |
| Ordinary validation | `src/extraction/compact_validation.py` | Check schema, IDs, links, field status, and evidence labels. | No |
| Candidate inventory | `src/extraction/build_outcome_candidates.py` + `consolidate_outcome_candidates.py` | For complex papers, build and deduplicate possible outcome groups from the full local evidence view. | No |
| Coverage | `src/extraction/check_outcome_coverage.py` | Match candidates to returned outcomes one-to-one. Unmatched candidates remain explicit. | No |
| Unified routing | `src/extraction/route_compact_findings.py` | Send invalid fields, omitted text records, omitted visual records, obsolete whole responses, and ambiguous candidates to distinct routes. | No |
| Field repair | `src/extraction/run_narrow_repair.py` | Correct one invalid field from bounded text evidence. | Yes, only if needed and confirmed |
| Missing text record | `src/extraction/run_missing_record_repair.py` | Add omitted experiments/outcomes; every candidate must be marked recovered or unresolved. | Yes, only if needed and confirmed |
| Missing visual record | `build_missing_record_vision_tasks.py` + `run_missing_record_vision.py` | Inspect one targeted page. Exact printed/derived values can proceed; visual estimates require human review. | Yes, only if needed and confirmed |
| Merge | `src/extraction/merge_missing_records.py` and `merge_compact_results.py` | Reject stale tasks, ID collisions, invented evidence, bad links, and unresolved candidates; never overwrite the source result. | No |
| Revalidation and coverage | same merge plus local validators | Re-run schema/evidence checks and complex-paper coverage after records are added. | No |
| Final benchmark | `src/extraction/evaluate_final_gold_dynamic.py` | Measure final one-to-one recovery from actual merged records without hard-coded recovered IDs. | No |

## Evidence views

Both views are the same `CompactApiPacket` type, are signed with the same
checksum rule, and are read by the same `load_packet`, so any consumer accepts
either one.

- **Compact view** (`data/staging/rag/compact_api_packets_v1/`): retrieval
  evidence ranked by priority and truncated to a 16,000-token budget. Frozen
  behavior; its prompt cache key is unchanged.
- **Full view** (`data/staging/rag/full_api_packets_v1/`): the same retrieval
  evidence rejoined with every locally parsed block. Evidence added by the
  corpus expansion is tagged only `outcomes` or `full_corpus` and carries no
  experiment anchors, so it is weaker than retrieval-tagged evidence.

Consequences that are enforced in code:

- The request fingerprint carries the view, so the two views never share a
  prompt cache key.
- `run_compact_one_call` estimates input size before sending and refuses any
  request above `--max-input-tokens` (default 150,000) instead of paying for it.
- The complexity census scales its volume-based thresholds for the full view and
  weights `full_corpus`-only passages at half, so corpus volume alone cannot
  route every paper to `complex`. Compact-view scoring is unchanged.

## Simple versus complex behavior

All papers receive the first call and ordinary validation.

- **Simple paper:** ordinary validation is sufficient. Only actual validation
  findings route to repair.
- **Complex paper:** ordinary validation still runs, then the full-corpus
  candidate inventory and one-to-one coverage check also run. A structurally
  valid response cannot finalize while a credible outcome candidate is
  unmatched or unadjudicated.

## Ordinary validation routes

- A repairable field error goes to bounded text repair or selective vision.
- A whole-response or obsolete-schema error goes to `first_call_required`.
- A valid simple response can finalize.
- A valid complex response continues to the coverage gate.

## Complex coverage routes

- Matched candidate: no second call.
- High-confidence unmatched text candidate: missing-record text task.
- High-confidence unmatched figure/table candidate: missing-record vision task.
- Medium-confidence or ambiguous candidate: human review before any paid call.
- A task response must account for every candidate ID; silence is not a valid
  disposition.

## Finalization rule

Finalization is allowed only when:

- ordinary validation is valid;
- every complex-paper candidate is matched, recovered, explicitly rejected by
  adjudication, or explicitly unresolved;
- no required repair, vision task, or human review remains;
- the merged records pass schema, evidence-ID, link, uniqueness, and coverage
  checks.

Candidate recall is not final recovery. On the current frozen gold set the
local inventory recalls 15/15 outcomes, while the current merged results
contain 10/15 (66.7%). GO-002, GO-003, GO-006, GO-017, and GO-018 remain
unrecovered; they must not be reported as already extracted.
