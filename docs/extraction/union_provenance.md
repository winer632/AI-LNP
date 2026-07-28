# Rebuilding `codex_union_v1`

`data/staging/extraction/codex_union_v1` is the recall-first ensemble that the
11/15 headline is measured on. It is not an extraction: every one of its 51
outcome records was produced and validated by one of five input runs, and each
record names its origin in `source_run`.

Until this document existed, three of those five roots lived only in a working
directory outside the repository, and `union_manifest.json` cited them by
absolute path:

```
/private/tmp/claude-.../scratchpad/repair_v2_merged
/private/tmp/claude-.../scratchpad/p1_exp/extraction_sc
/private/tmp/claude-.../scratchpad/p1_exp/extraction_p3_compact
```

35 of the 51 records came from those three roots, so a fresh clone could
evaluate the union but could not rebuild it. The runs are now committed under
their original directory names — the union writes `source_run` from the run
root's basename, so renaming them would have rewritten every record — and the
manifest cites the committed paths.

## Rebuild command

From a fresh clone, with no network access and no API key:

```bash
.venv/bin/python -m src.extraction.union_extraction_results \
  --run-root data/staging/extraction/repair_v2_merged \
  --run-root data/staging/extraction/extraction_sc \
  --run-root data/staging/extraction/extraction_p3_compact \
  --run-root data/staging/extraction/codex_treatment_full_v1 \
  --run-root data/staging/extraction/codex_control_compact_v1 \
  --output-root /tmp/union_rebuild
```

Root order is part of the definition, not a detail: the union keeps the first
record it sees for a given endpoint/value/unit key, so reordering the roots
changes which run is credited in `source_run`.

`tests/test_union_provenance.py` runs exactly this build and compares it against
the committed artifact.

### What the rebuild reproduces exactly

* `union_manifest.json` — byte for byte, including per-paper counts and
  `by_source_run`.
* Every outcome record's payload, in the same order, under the same
  `source_run`: same endpoint, value, unit, evidence ids, everything.
* The measured numbers: 11/15 recovered, precision 0.2157 over 51 records, 40
  false additions, evidence 6 supported of 10 checked.

### The one thing it does not reproduce

`outcome_id`. The committed artifact was written by the builder as of commit
57265dc. Commit cb3b3a7 then taught `union_extraction_results.py` to rename an
`outcome_id` that collides with one already taken — runs number their own
records from `O1`, so a union holds several records called `O1`, and anything
that groups by id conflates them. The artifact was never regenerated, so it
still holds 51 records under 32 distinct ids, while a rebuild today emits
`O1-2`, `O1-3`, … plus a `source_outcome_id` field recording the original.

This changes no measurement. `evaluate_final_gold_dynamic.py` selects records by
position, not by id, precisely because ids repeat on a union root, so the
rebuilt union scores identically — the test asserts that rather than asserting
it in prose. The artifact was left as-is because
`data/staging/extraction/codex_union_vision_v2` was derived from these exact
bytes and its build is not scripted in the repository; regenerating one without
the other would move the drift rather than remove it.

## Provenance table

| `source_run` | records | committed at | what it is |
|---|---:|---|---|
| `repair_v2_merged` | 20 | `data/staging/extraction/repair_v2_merged` | `codex_control_compact_v1` plus the missing-record repair fragments that resolved, merged by `merge_missing_records.py` |
| `codex_treatment_full_v1` | 14 | `data/staging/extraction/codex_treatment_full_v1` | full evidence view, one call |
| `extraction_sc` | 8 | `data/staging/extraction/extraction_sc` | structured-compact packet at a 16k budget, candidate slots enforced (53 slots) |
| `extraction_p3_compact` | 7 | `data/staging/extraction/extraction_p3_compact` | compact packet, candidate slots enforced (39 slots) |
| `codex_control_compact_v1` | 2 | `data/staging/extraction/codex_control_compact_v1` | compact packet control run |

Only GP-002 and GP-004 through GP-008 contribute records; GP-001, GP-003 and
GP-009 produce none in any input run.

### Which gold outcome each root actually carries

| gold | matched record's `source_run` |
|---|---|
| GO-001 | `repair_v2_merged` |
| GO-004 | `extraction_p3_compact` |
| GO-005 | `extraction_sc` |
| GO-006 | `extraction_sc` |
| GO-007 | `repair_v2_merged` |
| GO-008 | `codex_treatment_full_v1` |
| GO-010 | `repair_v2_merged` |
| GO-011 | `codex_treatment_full_v1` |
| GO-013 | `codex_treatment_full_v1` |
| GO-015 | `codex_treatment_full_v1` |
| GO-016 | `repair_v2_merged` |

GO-002, GO-003, GO-017 and GO-018 are recovered by no root.

Attribution is not the same as necessity. Matching is one-to-one over the whole
paper, so removing a root can make the matcher re-assign a gold row to a record
it had passed over. Measured by rebuilding the union from the two roots that
were already committed (`codex_treatment_full_v1` and
`codex_control_compact_v1`) and re-running the evaluator:

| union | recall | precision | records |
|---|---:|---:|---:|
| all five roots | 11/15 (0.7333) | 0.2157 | 51 |
| committed-only roots, before this change | 9/15 (0.6000) | 0.3462 | 26 |

Two gold outcomes are lost, not five: **GO-004** and **GO-006**. GO-001, GO-005,
GO-007, GO-010 and GO-016 survive because another root's record matches them
once the preferred one is gone. So the honest headline for a clone that lacked
the three roots was **9/15**, and restoring them is what makes 11/15 a number a
reader can rebuild rather than take on trust.

## Deeper provenance

The union's inputs are themselves derived artifacts, and their own inputs are
committed too.

`repair_v2_merged` was produced by `merge_missing_records.py` from:

* base result — `codex_control_compact_v1/<paper>/result.json` for all four
  papers, checked by the `source_result_sha256` each merge report records;
* repair tasks — `data/staging/extraction/repair_v2/<paper>/task_*.json`;
* model fragments — `data/staging/extraction/repair_v2_out/<paper>/…/fragment.json`,
  alongside the `request.json`, `response.json` and `manifest.json` of each call.

The merge reports still record the absolute working-directory paths those files
had when the merge ran. The mapping is a prefix substitution: everything under
`…/scratchpad/` sits under `data/staging/extraction/` with the same relative
path. `tests/test_union_provenance.py` checks that every recorded path resolves.

Ten of `repair_v2_merged`'s 20 records are byte-identical to records in
`codex_control_compact_v1`: the merge only ever appends, so every base record
comes through unchanged, and on GP-004 and GP-008 nothing was recovered at all.
The other ten — `OC-3bd6c98e48bb9894` on GP-005 and nine `OC-*` records on
GP-006 — exist in no other committed run, and they are why
`codex_control_merged_v1` is not a substitute for this root: that run recovered
2 of GP-006's 9 candidates, this one recovered all 9, and it does not cover
GP-005 or GP-008 at all.

`extraction_sc` and `extraction_p3_compact` are single codex-exec calls and
carry their own `request.json`, `response.json` and `manifest.json`. Each
request records the checksum of the packet it was sent:

| run | `packet_checksum` | packet |
|---|---|---|
| `extraction_sc` | `20cec003…` | `data/staging/rag/structured_compact_packets_sc_v1/GP-006.json` |
| `extraction_p3_compact` | `d556f6c4…` | `data/staging/rag/compact_api_packets_v1/GP-006.json` |

The structured-compact packet the `extraction_sc` run was sent is a 16k-budget
build distinct from `structured_compact_packets_v1`, and was committed with the
run. Its original path was `…/scratchpad/p1_exp/scpackets/`.

What is *not* reproducible from the repository is the model call itself: these
are LLM outputs, and re-running them would produce different text. That is true
of every extraction run this repository ships. What is reproducible is
everything downstream of the responses, which is what the union is.
