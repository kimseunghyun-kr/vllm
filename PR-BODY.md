<!-- markdownlint-disable -->

## Overview

`VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT` (30 s by default) also counts the time a chat-template render waits in the renderer pool queue. The pool is shared with prompt tokenization and decoding of all requests, and it has one thread by default (`renderer_num_workers = 1`). Under concurrent long prompts, a render that takes milliseconds can fail with the "malicious or excessively complex template" error (HTTP 500), and the pool is replaced. This PR counts only the render's own run time.

## Claims

- **Bug fix:** a render that waits in the pool queue for longer than the timeout no longer fails. The timeout starts when a worker starts the render.
- A render that is slow by itself still times out after the same limit and still replaces the pool, with the same error text.
- A render that is still queued when the pool is replaced (after another render got stuck) is submitted again to the new pool, so it does not wait behind the stuck thread. Today such a render also fails after the timeout.
- No change when `VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT <= 0`, and no change for renderers that do not use `_render_with_timeout`.

## Validation

**Reproducer (unpatched code, vLLM 5555a2d8 image, CPU only):** the renderer pool runs a 1 s job (standing in for the tokenization of a long prompt), then a render that returns at once is submitted with a timeout of 0.5 s.

```
REPRO TimeoutError after 0.50 s: Chat template rendering timed out after 0.5s. This may indicate a malicious or excessively complex template. Adjust VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT to override.
```

With this change the same case is the test `test_render_timeout_excludes_queue_wait`, and it passes.

**Tests** (`tests/renderers/test_executor_replace.py`, run in the same image with the patched `vllm/renderers/hf.py`):

```bash
pytest -q tests/renderers/test_executor_replace.py
```

- With the change: 5 passed (2 existing, 3 new).
- New tests:
  - `test_render_timeout_excludes_queue_wait`: the reproducer above (on `main` the same case raises TimeoutError, shown by the reproducer script; the test itself uses the new call signature).
  - `test_slow_render_still_times_out_and_replaces_pool`: a render that takes 1 s with a 0.3 s timeout still raises and still replaces the inner pool.
  - `test_render_queued_behind_stuck_render_moves_to_new_pool`: a fast render queued behind a stuck one succeeds after the stuck render times out. On `main` it would also time out (from the code; not run there).
- The tests run in the existing ":computer: (CPU) Reasoning + Renderers" CI step (`pytest -v -s renderers`).

**Serving observation (no fix applied, for context):** on 8 × A100-SXM4-40GB, Qwen3.8-Flash-Next-FP8, MLPerf agentic multi-turn workload with 32 sessions and prompts of up to about 250 k tokens, one 1-hour run had 3 such HTTP 500 errors per server arm. Each one aborted the rest of its conversation. For the same prompts, measured offline with the model's own template and tokenizer: render 0.1-4.3 ms (up to 379 messages, 777 k characters), tokenization 71-293 ms (42 k-249 k tokens). So the 30 s were queue time, not render time.

**Not measured:** no serving run with this change. `ruff` and the full `pre-commit` suite were not run locally; lines are within 88 characters.

## Details

**Root cause.** `HfRenderer._render_with_timeout(coro)` did `asyncio.wait_for(coro, timeout)`, where `coro` is the future from `make_async(safe_apply_chat_template, executor=self._executor)`. The future exists from the moment the job is queued, so the timeout clock started at submission. `self._executor` is the `_SwappableExecutor` that also runs `_tokenize_prompt_async`, `_async_tokenizer_decode` and the other renderer jobs.

**Fix.** `_render_with_timeout(*args, **kwargs)` now submits `safe_apply_chat_template` itself. The submitted function first signals the event loop that it started (`loop.call_soon_threadsafe` on a future). The coroutine waits, without a timeout, for either that signal or a pool replacement. After the start, it applies the timeout to the render future as before. If the pool is replaced first, the job is marked abandoned (it returns at once if the old pool ever runs it) and is submitted again to the new pool. `_replace_executor()` sets an `asyncio.Event` for that.

**Trade-offs and limits.**
- A render is never timed out while it waits in the queue. If the pool is busy for a long time with other work that never gets stuck, the render waits as long as that work takes. This is the same as tokenization and decoding today, which have no timeout.
- The added cost per render is one `call_soon_threadsafe` and one small task.
- Other renderers (DeepSeek V3.2/V4, Mistral, Kimi K3) call their own `_apply_chat_template_async` without this timeout and are not changed.
- Related: #50300 added the timeout and the pool replacement; #62374 fixed long-conversation HTTP 500s in the Rust frontend (a different cause, the instruction budget). No open PR or issue covers this case.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

AI assistance: Claude Code (Claude Opus 5.5) wrote the change, the tests and this description. The submitter ran the reproducer and the tests in the serving image, and reviews the change before it leaves draft.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
