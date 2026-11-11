# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-space sampling and verification against the full-vocabulary kernels.

Each case builds processed target logits with a few kept tokens per row and the
same rows as candidates. `gumbel_sample_candidates` and
`rejection_sample_candidates` must return the tokens of `gumbel_sample` and
`rejection_sample` (one-hot drafts, standard mode).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.candidates import (
    gumbel_sample_candidates,
    rejection_sample_candidates,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the sampling kernels need CUDA"
)

VOCAB_SIZE = 248320
NUM_CANDIDATES = 20
MAX_NUM_REQS = 512
LOGIT_SPREAD = 3.0
TOP_P = 0.95


def _kept_candidates(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k then top-p (vLLM's PyTorch rule). Candidates sorted by token id."""
    values, token_ids = logits.topk(NUM_CANDIDATES, dim=-1)
    mass_from_here = values.softmax(dim=-1).flip(-1).cumsum(dim=-1).flip(-1)
    dropped = mass_from_here <= 1.0 - TOP_P
    dropped[:, 0] = False
    values = values.masked_fill(dropped, float("-inf"))
    token_ids, order = token_ids.sort(dim=-1)
    return values.gather(1, order).contiguous(), token_ids.contiguous()


class Batch:
    """Requests with 0..num_steps draft tokens each, in rejection sampler layout."""

    def __init__(self, num_reqs: int, num_steps: int, greedy_share: float, seed: int):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(seed)

        def uniform(*shape):
            return torch.rand(*shape, generator=generator, device=device)

        def integers(high, *shape):
            return torch.randint(0, high, shape, generator=generator, device=device)

        self.num_steps = num_steps
        num_drafts = integers(num_steps + 1, num_reqs)
        self.cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        self.cu_num_logits[1:] = (num_drafts + 1).cumsum(0)
        num_logits = int(self.cu_num_logits[-1])
        request_of_row = torch.repeat_interleave(
            torch.arange(num_reqs, device=device), num_drafts + 1
        )
        self.idx_mapping = torch.randperm(
            MAX_NUM_REQS, generator=generator, device=device
        )[:num_reqs].to(torch.int32)
        self.expanded_idx_mapping = self.idx_mapping[request_of_row].contiguous()
        starts = self.cu_num_logits[:-1].to(torch.int64)[request_of_row]
        rows = torch.arange(num_logits, device=device)
        self.expanded_local_pos = (rows - starts).to(torch.int32)
        self.pos = 1000 + integers(1 << 20, num_logits)
        self.temperature = torch.ones(MAX_NUM_REQS, device=device)
        self.temperature[uniform(MAX_NUM_REQS) < greedy_share] = 0.0
        self.seed = integers(1 << 62, MAX_NUM_REQS)

        raw = torch.randn(
            num_logits, VOCAB_SIZE, generator=generator, device=device
        ).mul_(LOGIT_SPREAD)
        self.values, self.token_ids = _kept_candidates(raw)
        is_greedy_row = self.temperature[self.expanded_idx_mapping.long()] == 0.0
        # Greedy rows have no top-k or top-p mask: every candidate stays.
        unmasked = raw.gather(1, self.token_ids)
        self.values = torch.where(is_greedy_row[:, None], unmasked, self.values)
        self.target_logits = torch.full_like(raw, float("-inf"))
        self.target_logits.scatter_(1, self.token_ids, self.values)
        self.target_logits = torch.where(
            is_greedy_row[:, None], raw, self.target_logits
        )

        # Draft token of a row: the best token, another candidate, a token
        # outside the candidates, or the -1 placeholder.
        best = raw.argmax(dim=-1)
        other = self.token_ids.gather(1, integers(NUM_CANDIDATES, num_logits, 1))[:, 0]
        outside = integers(VOCAB_SIZE, num_logits)
        choice = uniform(num_logits)
        draft = torch.where(choice < 0.6, best, other)
        draft = torch.where(choice > 0.8, outside, draft)
        draft = torch.where(choice > 0.95, torch.full_like(draft, -1), draft)
        # The draft token that row r verifies is stored at row r + 1.
        self.draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=device)
        self.draft_sampled[1:] = draft[:-1]
        self.draft_sampled[self.cu_num_logits[:-1].long()] = 0


def _valid_prefix_equal(sampled, num_sampled, expected, expected_num) -> torch.Tensor:
    columns = torch.arange(sampled.shape[1], device=sampled.device)[None, :]
    within = columns < num_sampled[:, None]
    same_tokens = ((sampled == expected) | ~within).all(dim=-1)
    return same_tokens & (num_sampled == expected_num)


@pytest.mark.parametrize("greedy_share", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("num_steps", [1, 3])
def test_rejection_sample_candidates_matches_full_vocabulary(greedy_share, num_steps):
    mismatched = total = accepted_drafts = 0
    for seed in range(6):
        batch = Batch(64, num_steps, greedy_share, seed)
        expected, expected_num = rejection_sample(
            batch.target_logits,
            None,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            batch.expanded_idx_mapping,
            batch.expanded_local_pos,
            batch.temperature,
            batch.seed,
            num_steps,
        )
        sampled, num_sampled = rejection_sample_candidates(
            batch.values,
            batch.token_ids,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            batch.temperature,
            batch.seed,
            num_steps,
        )
        equal = _valid_prefix_equal(sampled, num_sampled, expected, expected_num)
        mismatched += int((~equal).sum())
        total += equal.numel()
        accepted_drafts += int((expected_num - 1).sum())
    print(
        f"requests {total}, mismatched {mismatched}, accepted drafts {accepted_drafts}"
    )
    # The test must exercise both acceptance and rejection.
    assert 0 < accepted_drafts < total * num_steps
    assert mismatched == 0


@pytest.mark.parametrize("greedy_share", [0.0, 0.5])
def test_gumbel_sample_candidates_matches_full_vocabulary(greedy_share):
    for seed in range(4):
        batch = Batch(96, 0, greedy_share, seed)
        expected = gumbel_sample(
            batch.target_logits,
            batch.expanded_idx_mapping,
            batch.temperature,
            batch.seed,
            batch.pos,
            apply_temperature=False,
            is_drafting=False,
        )
        sampled = gumbel_sample_candidates(
            batch.values,
            batch.token_ids,
            batch.expanded_idx_mapping,
            batch.temperature,
            batch.seed,
            batch.pos,
        )
        assert torch.equal(sampled, expected)


def test_report_kernel_times():
    """Not an assertion: print the kernel times at 128 logits rows (32 requests x 4)."""
    import statistics
    import time

    batch = Batch(32, 3, 0.0, 0)
    batch.cu_num_logits = torch.arange(0, 129, 4, dtype=torch.int32, device="cuda")
    rows = 128
    for name in ("target_logits", "values", "token_ids", "draft_sampled", "pos"):
        tensor = getattr(batch, name)
        repeats = -(-rows // tensor.shape[0])
        setattr(
            batch,
            name,
            tensor.repeat(repeats, *[1] * (tensor.ndim - 1))[:rows].contiguous(),
        )
    batch.expanded_idx_mapping = batch.idx_mapping.repeat_interleave(4).contiguous()
    batch.expanded_local_pos = torch.arange(4, dtype=torch.int32, device="cuda").repeat(
        32
    )

    def full():
        return rejection_sample(
            batch.target_logits,
            None,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            batch.expanded_idx_mapping,
            batch.expanded_local_pos,
            batch.temperature,
            batch.seed,
            3,
        )

    def candidates():
        return rejection_sample_candidates(
            batch.values,
            batch.token_ids,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            batch.temperature,
            batch.seed,
            3,
        )

    for name, call in (
        ("rejection_sample", full),
        ("rejection_sample_candidates", candidates),
    ):
        for _ in range(5):
            call()
        samples = []
        for _ in range(30):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        print(f"{name}: {statistics.median(samples) * 1e6:.0f} us at {rows} rows")
    expected, expected_num = full()
    sampled, num_sampled = candidates()
    assert bool(_valid_prefix_equal(sampled, num_sampled, expected, expected_num).all())


NUM_SHARDS = 8
SHARD_CANDIDATES = 32


class ProcessingCase:
    """Raw bf16 target logits (with ties) and per-request sampling state."""

    def __init__(self, seed: int):
        from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
        from vllm.v1.worker.gpu.sample.penalties import apply_penalties

        self.apply_temperature, self.apply_penalties = (
            apply_temperature,
            apply_penalties,
        )
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(1000 + seed)
        self.batch = batch = Batch(48, 3, 0.2, 100 + seed)

        def uniform(*shape):
            return torch.rand(*shape, generator=generator, device=device)

        num_logits = batch.target_logits.shape[0]
        self.raw = (
            torch.randn(num_logits, VOCAB_SIZE, generator=generator, device=device)
            .mul_(LOGIT_SPREAD)
            .to(torch.bfloat16)
        )
        greedy = batch.temperature == 0.0
        choice = uniform(MAX_NUM_REQS)
        batch.temperature = torch.where(
            ~greedy & (choice < 0.3), torch.full_like(choice, 0.7), batch.temperature
        )
        self.top_k = torch.full((MAX_NUM_REQS,), 20, dtype=torch.int32, device=device)
        self.top_k[uniform(MAX_NUM_REQS) < 0.2] = 5
        self.top_k[greedy] = VOCAB_SIZE
        self.top_p = torch.full((MAX_NUM_REQS,), TOP_P, device=device)
        self.top_p[uniform(MAX_NUM_REQS) < 0.2] = 0.5
        self.top_p[uniform(MAX_NUM_REQS) < 0.2] = 1.0
        self.top_p[greedy] = 1.0
        self.repetition_penalty = torch.ones(MAX_NUM_REQS, device=device)
        self.repetition_penalty[uniform(MAX_NUM_REQS) < 0.3] = 1.1
        self.frequency_penalty = torch.zeros(MAX_NUM_REQS, device=device)
        self.frequency_penalty[uniform(MAX_NUM_REQS) < 0.3] = 0.2
        self.presence_penalty = torch.zeros(MAX_NUM_REQS, device=device)
        self.presence_penalty[uniform(MAX_NUM_REQS) < 0.7] = 1.5
        self.prompt_bin_mask = torch.randint(
            -(2**31),
            2**31,
            (MAX_NUM_REQS, VOCAB_SIZE // 32),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(torch.int32)
        self.prompt_bin_mask &= self.prompt_bin_mask.roll(1, 1)  # about 1/4 of the bits
        self.output_bin_counts = (uniform(MAX_NUM_REQS, VOCAB_SIZE) < 0.01).to(
            torch.int32
        )
        # Some of each row's best tokens were already generated by its request.
        best = self.raw.float().topk(40, dim=-1).indices[:, ::3]
        requests = batch.expanded_idx_mapping.long()[:, None].expand_as(best)
        self.output_bin_counts.index_put_(
            (requests, best), torch.ones_like(best, dtype=torch.int32)
        )

    def _process(self, logits, prompt_bin_mask, output_bin_counts, vocab_offset):
        batch = self.batch
        self.apply_penalties(
            logits,
            batch.expanded_idx_mapping,
            batch.draft_sampled,
            batch.expanded_local_pos,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
            prompt_bin_mask,
            output_bin_counts,
            vocab_offset=vocab_offset,
        )
        self.apply_temperature(logits, batch.expanded_idx_mapping, batch.temperature)

    def full_vocabulary(self) -> torch.Tensor:
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

        logits = torch.empty_like(self.raw, dtype=torch.float32).copy_(self.raw)
        self._process(logits, self.prompt_bin_mask, self.output_bin_counts, 0)
        requests = self.batch.expanded_idx_mapping.long()
        return apply_top_k_top_p(
            logits, self.top_k[requests], self.top_p[requests], has_topp_only_rows=False
        )

    def candidates(self) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.v1.worker.gpu.sample.candidates import (
            finalize_candidates,
            merge_shard_candidates,
            top_k_shard_candidates,
        )

        width = VOCAB_SIZE // NUM_SHARDS
        packed = []
        for start in range(0, VOCAB_SIZE, width):
            shard = self.raw[:, start : start + width]
            logits = torch.empty_like(shard, dtype=torch.float32).copy_(shard)
            self._process(
                logits,
                self.prompt_bin_mask[:, start // 32 : (start + width) // 32],
                self.output_bin_counts[:, start : start + width],
                start,
            )
            packed.append(top_k_shard_candidates(logits, start, SHARD_CANDIDATES))
        values, token_ids = merge_shard_candidates(
            torch.cat(packed, dim=-1), NUM_SHARDS
        )
        return finalize_candidates(
            values, token_ids, self.batch.expanded_idx_mapping, self.top_k, self.top_p
        )


def test_shard_candidates_match_full_vocabulary_processing():
    """Per-shard processing + top-k + merge against the full-vocabulary chain
    (penalties, temperature, top-k, top-p), then sampling and verification."""
    rows = bad_rows = bad_requests = requests_total = 0
    for seed in range(4):
        case = ProcessingCase(seed)
        batch = case.batch
        expected = case.full_vocabulary()
        values, token_ids = case.candidates()
        is_greedy_row = batch.temperature[batch.expanded_idx_mapping.long()] == 0.0
        rebuilt = torch.full_like(expected, float("-inf")).scatter_(
            1, token_ids, values
        )
        # A greedy row keeps its whole vocabulary: compare it on the candidates.
        rebuilt = torch.where(is_greedy_row[:, None], expected, rebuilt)
        on_candidates = expected.gather(1, token_ids)
        same = (rebuilt == expected).all(dim=-1) & (
            ~is_greedy_row | (on_candidates == values).all(dim=-1)
        )
        same &= ~is_greedy_row | (
            values.max(dim=-1).values == expected.max(dim=-1).values
        )
        rows += same.numel()
        bad_rows += int((~same).sum())

        full_tokens, full_num = rejection_sample(
            expected,
            None,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            batch.expanded_idx_mapping,
            batch.expanded_local_pos,
            batch.temperature,
            batch.seed,
            batch.num_steps,
        )
        tokens, num = rejection_sample_candidates(
            values,
            token_ids,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            batch.temperature,
            batch.seed,
            batch.num_steps,
        )
        equal = _valid_prefix_equal(tokens, num, full_tokens, full_num)
        requests_total += equal.numel()
        bad_requests += int((~equal).sum())
        kept = (expected > float("-inf")).sum(-1)[~is_greedy_row].float()
        print(f"seed {seed}: kept per row mean {kept.mean():.2f} max {kept.max():.0f}")
    print(
        f"rows {rows}, rows with a different kept set or value {bad_rows}; "
        f"requests {requests_total}, requests with different tokens {bad_requests}"
    )
    assert bad_rows == 0
    assert bad_requests == 0
