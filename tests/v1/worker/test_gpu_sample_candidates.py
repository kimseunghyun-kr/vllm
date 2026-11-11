# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-space sampling and verification against the full-vocabulary kernels.

`gumbel_sample_candidates` and `rejection_sample_candidates` must return the
tokens of `gumbel_sample` and `rejection_sample` (one-hot drafts, standard
mode), and keep the tokens that `apply_top_k_top_p` keeps.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.candidates import (
    _kept_candidates,
    gumbel_sample_candidates,
    rejection_sample_candidates,
    shard_top_k_supported,
    top_k_shard_candidates,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the sampling kernels need CUDA"
)

VOCAB_SIZE = 248320
NUM_CANDIDATES = 20
NUM_SHARDS = 8
MERGED_CANDIDATES = 32  # candidates._MERGED_CANDIDATES
MAX_NUM_REQS = 512
LOGIT_SPREAD = 3.0
TOP_P = 0.95


def _reference_kept(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k then top-p (vLLM's PyTorch rule): values (-inf if dropped), ids."""
    values, token_ids = logits.topk(NUM_CANDIDATES, dim=-1)
    mass_from_here = values.softmax(dim=-1).flip(-1).cumsum(dim=-1).flip(-1)
    dropped = mass_from_here <= 1.0 - TOP_P
    dropped[:, 0] = False
    return values.masked_fill(dropped, float("-inf")), token_ids


def _one_shard(values: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gathered layout of one shard that holds these candidates."""
    return torch.cat([values, token_ids.to(values.dtype)], dim=-1).contiguous()


def _keep_all(batch: "Batch") -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k and top-p state that keeps every candidate of `batch`."""
    top_k = torch.full((MAX_NUM_REQS,), NUM_CANDIDATES, dtype=torch.int32)
    return top_k.to(batch.pos.device), torch.ones(MAX_NUM_REQS, device=batch.pos.device)


@triton.jit
def _kept_candidates_kernel(
    values_ptr,
    token_ids_ptr,
    gathered_ptr,
    gathered_stride,
    expanded_idx_mapping_ptr,
    top_k_ptr,
    top_p_ptr,
    num_shards,
    num_candidates,
    PADDED_SHARDS: tl.constexpr,
    PADDED_CANDIDATES: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    values, token_ids, valid = _kept_candidates(
        gathered_ptr,
        gathered_stride,
        logit_idx,
        tl.load(top_k_ptr + req_state_idx),
        tl.load(top_p_ptr + req_state_idx).to(tl.float32),
        num_shards,
        num_candidates,
        PADDED_SHARDS,
        PADDED_CANDIDATES,
    )
    SIZE: tl.constexpr = values.shape[0]
    out = logit_idx * SIZE + tl.arange(0, SIZE)
    tl.store(values_ptr + out, values)
    tl.store(token_ids_ptr + out, tl.where(valid, token_ids, 0))


def _kept_logits(
    gathered: torch.Tensor,
    num_shards: int,
    expanded_idx_mapping: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """`[num_logits, vocab_size]` fp32: the kept candidates, -inf elsewhere."""
    num_candidates = gathered.shape[1] // (2 * num_shards)
    shape = (gathered.shape[0], MERGED_CANDIDATES)
    values = gathered.new_empty(shape)
    token_ids = gathered.new_empty(shape, dtype=torch.int64)
    _kept_candidates_kernel[(gathered.shape[0],)](
        values,
        token_ids,
        gathered,
        gathered.stride(0),
        expanded_idx_mapping,
        top_k,
        top_p,
        num_shards,
        num_candidates,
        PADDED_SHARDS=triton.next_power_of_2(num_shards),
        PADDED_CANDIDATES=triton.next_power_of_2(num_candidates),
    )
    kept = gathered.new_full((gathered.shape[0], vocab_size), float("-inf"))
    return kept.scatter_reduce_(1, token_ids, values, reduce="amax")


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
        self.values, self.token_ids = _reference_kept(raw)
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
            _one_shard(batch.values, batch.token_ids),
            1,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            *_keep_all(batch),
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
            _one_shard(batch.values, batch.token_ids),
            1,
            batch.expanded_idx_mapping,
            *_keep_all(batch),
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

    gathered = _one_shard(batch.values, batch.token_ids).repeat(1, NUM_SHARDS)
    # Distinct token ids on the other shards, with values that are never kept.
    gathered = gathered.view(rows, NUM_SHARDS, 2, NUM_CANDIDATES).clone()
    gathered[:, 1:, 0] = float("-inf")
    gathered[:, 1:, 1] += VOCAB_SIZE * torch.arange(
        1, NUM_SHARDS, device=gathered.device
    ).view(1, -1, 1)
    gathered = gathered.view(rows, -1)
    top_k, top_p = _keep_all(batch)

    def candidates():
        return rejection_sample_candidates(
            gathered,
            NUM_SHARDS,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            top_k,
            top_p,
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


class ProcessingCase:
    """Raw bf16 target logits and per-request sampling state. `logit_step`
    rounds the logits to its multiples, so that many tokens tie."""

    def __init__(self, seed: int, logit_step: float = 0.0):
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
        raw = torch.randn(
            num_logits, VOCAB_SIZE, generator=generator, device=device
        ).mul_(LOGIT_SPREAD)
        if logit_step:
            raw = raw.div_(logit_step).round_().mul_(logit_step)
        self.raw = raw.to(torch.bfloat16)
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

    def gathered_candidates(self) -> torch.Tensor:
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
            packed.append(top_k_shard_candidates(logits, start, NUM_CANDIDATES))
        return torch.cat(packed, dim=-1)


needs_shard_top_k = pytest.mark.skipif(
    not shard_top_k_supported(), reason="needs FlashInfer top-k with a tie-break"
)


@needs_shard_top_k
def test_shard_top_k_prefers_lower_token_ids():
    """Among equal logits at the selection boundary, the lower columns win."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(3)
    width, vocab_start = VOCAB_SIZE // NUM_SHARDS, 64
    logits = torch.randint(0, 6, (96, width), generator=generator, device=device)
    logits = logits.float()
    logits[0] = 1.0  # every column ties
    packed = top_k_shard_candidates(logits, vocab_start, NUM_CANDIDATES)
    values = packed[:, :NUM_CANDIDATES]
    columns = packed[:, NUM_CANDIDATES:].long() - vocab_start
    expected = logits.sort(dim=-1, descending=True, stable=True)
    assert torch.equal(
        columns.sort(dim=-1).values,
        expected.indices[:, :NUM_CANDIDATES].sort(dim=-1).values,
    )
    assert torch.equal(values, logits.gather(1, columns))


@needs_shard_top_k
@pytest.mark.parametrize("logit_step", [0.0, 0.5])
def test_shard_candidates_match_full_vocabulary_processing(logit_step):
    """Per-shard processing + top-k, then the fused kernels, against the
    full-vocabulary chain (penalties, temperature, top-k, top-p, sampling and
    verification)."""
    rows = bad_rows = bad_samples = bad_requests = requests_total = 0
    over_kept_rows = 0
    for seed in range(4):
        case = ProcessingCase(seed, logit_step)
        batch = case.batch
        expected = case.full_vocabulary()
        gathered = case.gathered_candidates()
        is_greedy_row = batch.temperature[batch.expanded_idx_mapping.long()] == 0.0
        kept = _kept_logits(
            gathered,
            NUM_SHARDS,
            batch.expanded_idx_mapping,
            case.top_k,
            case.top_p,
            VOCAB_SIZE,
        )
        # A greedy row keeps its whole vocabulary: compare its best logit.
        same = torch.where(
            is_greedy_row,
            kept.max(dim=-1).values == expected.max(dim=-1).values,
            (kept == expected).all(dim=-1),
        )
        # With ties at the top-k boundary, apply_top_k_top_p can keep more than
        # top-k tokens. Such a row cannot match.
        requests = batch.expanded_idx_mapping.long()
        over_kept = (expected > float("-inf")).sum(dim=-1) > case.top_k[requests]
        over_kept &= ~is_greedy_row
        for row in (~same & ~over_kept).nonzero()[:4, 0].tolist():
            print(
                f"row {row}: top_k {int(case.top_k[requests[row]])} "
                f"top_p {float(case.top_p[requests[row]]):.2f} "
                f"kept {int((kept[row] > float('-inf')).sum())} "
                f"expected {int((expected[row] > float('-inf')).sum())}"
            )
        same |= over_kept
        over_kept_rows += int(over_kept.sum())
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
            gathered,
            NUM_SHARDS,
            batch.draft_sampled,
            batch.cu_num_logits,
            batch.pos,
            batch.idx_mapping,
            case.top_k,
            case.top_p,
            batch.temperature,
            batch.seed,
            batch.num_steps,
        )
        equal = _valid_prefix_equal(tokens, num, full_tokens, full_num)
        row_starts = batch.cu_num_logits[:-1].long()
        request_over_kept = torch.stack(
            [
                over_kept[start:stop].any()
                for start, stop in zip(row_starts, batch.cu_num_logits[1:].long())
            ]
        )
        requests_total += equal.numel()
        bad_requests += int((~equal & ~request_over_kept).sum())
        sampled = gumbel_sample_candidates(
            gathered,
            NUM_SHARDS,
            batch.expanded_idx_mapping,
            case.top_k,
            case.top_p,
            batch.temperature,
            batch.seed,
            batch.pos,
        )
        full_sampled = gumbel_sample(
            expected,
            batch.expanded_idx_mapping,
            batch.temperature,
            batch.seed,
            batch.pos,
            apply_temperature=False,
            is_drafting=False,
        )
        bad_samples += int(((sampled != full_sampled) & ~over_kept).sum())
        num_kept = (expected > float("-inf")).sum(-1)[~is_greedy_row].float()
        print(
            f"step {logit_step} seed {seed}: kept per row mean {num_kept.mean():.2f} "
            f"max {num_kept.max():.0f}"
        )
    print(
        f"rows {rows}, rows where the full path keeps more than top-k "
        f"{over_kept_rows}, other rows with a different kept set or value {bad_rows}, "
        f"with a different sample {bad_samples}; "
        f"requests {requests_total}, requests with different tokens {bad_requests}"
    )
    assert logit_step or over_kept_rows == 0
    assert bad_rows == 0
    assert bad_samples == 0
    assert bad_requests == 0


def _make_sampler(vocab_size: int, max_num_reqs: int):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=vocab_size,
        device=device,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=None),
        max_num_reqs=max_num_reqs,
        vocab_size=vocab_size,
        device=device,
        req_states=req_states,
    )


@needs_shard_top_k
def test_sampler_shard_candidates_match_apply_sampling_params(monkeypatch):
    """`Sampler.shard_candidates` on one shard that holds the whole vocabulary
    against `Sampler.apply_sampling_params`."""
    import numpy as np

    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample import sampler as sampler_module

    monkeypatch.setattr(
        sampler_module, "tensor_model_parallel_all_gather", lambda x, dim=-1: x
    )
    monkeypatch.setattr(
        sampler_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    vocab_size = 4096
    params = [
        SamplingParams(temperature=1.0, top_k=20, top_p=0.95),
        SamplingParams(temperature=0.7, top_k=5, top_p=0.5),
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.3, top_k=20),
        SamplingParams(temperature=1.0, top_k=1, top_p=0.9),
        SamplingParams(temperature=0.6, top_k=20, top_p=0.8),
    ]
    sampler = _make_sampler(vocab_size, len(params))
    for req_idx, sampling_params in enumerate(params):
        sampler.add_request(req_idx, sampling_params)
    sampler.apply_staged_writes()
    idx_mapping_np = np.arange(len(params), dtype=np.int32)
    assert sampler.supports_candidates(idx_mapping_np)

    device = torch.device("cuda")
    idx_mapping = torch.from_numpy(idx_mapping_np).to(device)
    generator = torch.Generator(device=device).manual_seed(5)
    logits = (
        torch.randn(len(params), vocab_size, generator=generator, device=device)
        .mul_(LOGIT_SPREAD)
        .to(torch.bfloat16)
    )
    context = (
        idx_mapping,  # expanded_idx_mapping: one logits row per request
        idx_mapping,
        idx_mapping_np,
        torch.arange(100, 100 + len(params), device=device),  # pos
        torch.zeros(len(params), dtype=torch.int32, device=device),  # input_ids
        torch.zeros(len(params), dtype=torch.int32, device=device),  # local pos
        np.ones(len(params), dtype=np.int32),
    )
    expected = sampler.apply_sampling_params(logits, *context)
    gathered = sampler.shard_candidates(logits, 0, *context)
    kept = _kept_logits(
        gathered,
        1,
        idx_mapping,
        sampler.sampling_states.top_k.gpu,
        sampler.sampling_states.top_p.gpu,
        vocab_size,
    )
    is_greedy = torch.tensor([p.temperature == 0.0 for p in params], device=device)
    expected = expected.float()
    assert torch.equal(kept[~is_greedy], expected[~is_greedy])
    assert torch.equal(kept[is_greedy].argmax(-1), expected[is_greedy].argmax(-1))

    sampler.add_request(
        0, SamplingParams(temperature=1.0, top_k=20, logit_bias={1: 2.0})
    )
    assert not sampler.supports_candidates(idx_mapping_np)
    sampler.add_request(0, SamplingParams(temperature=1.0, top_p=0.9))
    assert not sampler.supports_candidates(idx_mapping_np)
