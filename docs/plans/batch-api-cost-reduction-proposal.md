# Proposal: Claude Batch API for Generator Cost Reduction

**Status:** Draft / proposal
**Author:** Neil Berget (with Claude)
**Date:** 2026-06-08
**Related:** [clue-quality-and-cost-plan.md](clue-quality-and-cost-plan.md), `src/crossword_generator/llm/claude_provider.py`, `src/crossword_generator/llm/costs.py`

## TL;DR

The Anthropic **Message Batches API** processes Messages-API requests asynchronously at **50% of standard token prices**. With Opus 4.8 at $15/$75 per Mtok, clue generation and clue grading are our largest line items, so a 50% discount on those is material.

It is **not** a drop-in swap for the current per-puzzle pipeline — that pipeline is a data-dependent chain (a clue prompt doesn't exist until the grid is filled). Capturing the savings requires re-shaping the pipeline into **phase-batched** stages: fill every grid first, then submit all clue-generation requests as one batch, then all clue-grading requests as a second batch.

**Recommendation:** Pursue it for offline/scheduled generator runs, where the up-to-24h latency window is free. Build it behind a `--use-batch-api` flag so the existing synchronous path stays the default.

## Background

### How the Batch API works

- Endpoint: `POST /v1/messages/batches`. Already supported by our pinned `anthropic` SDK (`client.messages.batches.create / retrieve / results`) — no new dependency.
- Up to **100,000 requests** or **256 MB** per batch.
- Most batches finish **within 1 hour; hard cap 24 hours**. No latency SLA.
- **50% off** all token usage — input, output, cache creation, and cache read.
- Results retained 29 days; each request carries a `custom_id` to map results back to inputs.
- **Every Messages-API feature carries over**: tools, vision, thinking, `effort`, and **prompt caching**. Batch discount *stacks* with cache discounts.

### How our generator calls Claude today

The pipeline (`pipeline.py`) is five sequential steps per puzzle:

1. Theme generation (midi only) — Claude
2. Grid autofill (CSP) — **local, no Claude, cheap**
3. Fill grading — local word-list scoring
4. **Clue generation** — Claude, one call per puzzle (writes all entries)
5. **Clue grading** — Claude, with an internal retry/repair loop (`steps/clue_grading_step.py`)

On the `parallel-batch-execution` branch, `generate_pilot_batch` (`cli.py`) already flattens all `(bucket, seed)` puzzles into one work-list and runs **whole puzzles concurrently** across a thread pool (`--max-workers`). The only shared state is the thread-safe `ClueHistoryIndex`.

### Why it's not a drop-in replacement

The Batch API needs every request's full prompt **known up front**. Our calls are a dependent chain:

- The clue-generation prompt (step 4) depends on the **filled grid** produced by the CSP filler (step 2).
- The clue-grading prompt (step 5) depends on the **clues** produced by step 4.
- Step 5 then has an **internal repair loop** that issues *follow-up* Claude calls based on which clues were flagged — inherently iterative.

So within a single puzzle the Claude calls are a latency-sensitive chain — exactly what batch is *not* for. Our current thread-pool concurrency is the correct model for that shape and already gives us throughput.

## Where the savings actually live

The win comes from the one stage that is **embarrassingly parallel and fully knowable once grids exist**: clue generation across a *whole batch run*.

Restructure from a per-puzzle pipeline into **phase-batched** stages:

```
Phase A  Fill ALL grids        (local CSP — already cheap, run concurrently)
Phase B  Batch clue-generation (one Anthropic batch: N puzzles, known up front)  ← 50% off
Phase C  Batch clue-grading    (second batch over Phase B output)                 ← 50% off
Phase D  Repair pass           (synchronous, only for flagged clues — small tail)
```

- **Phase B** is the biggest line item (Opus 4.8 output tokens) and parallelizes perfectly — every grid is independent and complete before B starts.
- **Phase C** likewise: every clue set from B is gradeable independently.
- **Phase D** (the existing repair loop) stays synchronous. It touches only the minority of flagged clues, so it's a small tail and not worth batching.

This keeps the stateless `PuzzleEnvelope`-on-disk contract intact: each phase reads envelopes from disk, submits a batch, and writes envelopes back keyed by `custom_id`.

## Cost model (illustrative)

Current Opus 4.8 standard rates (`costs.py`): **$15 input / $75 output** per Mtok. Batch halves both to **$7.50 / $37.50**.

For a run of *P* puzzles where clue-gen + clue-grading dominate spend:

| Stage | Standard | Batch (50%) |
|---|---|---|
| Clue generation | full | **−50%** |
| Clue grading (first pass) | full | **−50%** |
| Repair tail (Phase D) | full | full (small) |

Net effect on a run is roughly a **40–50% reduction** on the LLM bill, depending on what fraction of spend is the repair tail. Prompt caching (already enabled on the system block, `claude_provider.py:93`) stacks on top — verify current cache hit rate first via the existing cost report (`crossword-generator cost-report`, `cli.py:92`).

> **Action item before committing:** run one representative batch and read the per-step cost report to confirm clue-gen + grading really are the dominant spend (expected, but measure).

## Tradeoffs

| | |
|---|---|
| **Pro** | Flat 50% discount; stacks with caching; no new dependency; fits our stateless envelope design. |
| **Pro** | Offline/scheduled runs (overnight data-store fills) don't care about the 24h window — the latency cost is free. |
| **Con** | Latency: batches can take up to 24h. Bad if a human is ever waiting on a run. |
| **Con** | Requires pipeline restructure (fill-all → batch → batch), plus submit/poll/collect plumbing and `custom_id` reassembly. |
| **Con** | Repair loop (Phase D) can't be batched — it stays synchronous and partially offsets savings. |
| **Con** | Error handling is per-request (`succeeded` / `errored` / `expired`) — needs resubmit logic for the tail. |

## Proposed implementation

Behind a `--use-batch-api` flag on `generate-pilot-batch`; the synchronous path remains the default.

1. **`llm/claude_batch.py`** — thin module wrapping `client.messages.batches`: `submit(requests) -> batch_id`, `poll(batch_id)` (with backoff), `collect(batch_id) -> dict[custom_id, LLMCallResponse]`. Reuse `_extract_text_content`, `_extract_usage`, and `estimate_llm_cost` from the existing provider so logging/cost accounting is identical.
2. **Phase split in `cli.py`** — when `--use-batch-api`:
   - Run Phase A (fill all grids) across the existing thread pool; persist envelopes through the fill+fill-grading steps.
   - Build clue-generation requests with `custom_id = f"cluegen:{difficulty}:{size}:seed-{seed}"`; submit one batch; collect; write clues back.
   - Build clue-grading requests the same way; submit; collect; write grades back.
   - Run Phase D repair synchronously for flagged clues only.
3. **Cost reporting** — batch results carry usage; record batch vs. standard pricing in the log records so `cost-report` shows the realized discount. (Add a `batch_discount` flag to `estimate_llm_cost`, or a separate batch rate table.)
4. **Resumability** — persist the batch IDs in the run manifest so an interrupted run can re-attach to an in-flight batch instead of resubmitting.

## Open questions

- Does `ClueHistoryIndex` duplicate-avoidance still work when all clue-gen requests are submitted *simultaneously* rather than sequentially? Today later puzzles see earlier puzzles' clues; in a single batch they won't. **This is the main correctness risk** — may need to accept slightly more cross-puzzle clue overlap within a batch, or run de-dup as a post-pass.
- Is the repair tail (Phase D) large enough to be worth a *second, smaller* batch instead of synchronous calls?
- Minimum viable first step: ship `llm/claude_batch.py` and run clue-generation-only through it manually to measure the real cost delta before committing to the full phase restructure.

## Recommendation

1. **First**: build `llm/claude_batch.py` and validate the cost delta on clue-generation alone (low risk, no pipeline change).
2. **If validated**: implement the phase-batched mode behind `--use-batch-api` for offline/scheduled runs.
3. **Keep** the synchronous thread-pool path as the default for any interactive or latency-sensitive run.
