# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling and speculative verification on per-shard top-k candidates.

With top-k sampling, the sampled token depends only on the k best processed
logits of a row. Each tensor-parallel rank sends the k best `(value, token id)`
pairs of its vocabulary shard instead of the shard's logits. One kernel then
merges the shards (a bitonic top-k of sortable keys), applies top-k and top-p,
and samples or verifies. It draws the same uniforms and the same Gumbel noise
as `gumbel_sample` and `rejection_sample`, because both are keyed by seed,
position and token id.

Candidates are ordered by value, ties by the lower token id, on each shard and
in the merge: the order of `apply_top_k_top_p`. The kept tokens are those of
the full-vocabulary path for every top-k up to the candidates per shard, with
one exception. When the top-p cut falls inside a group of equal logits,
`apply_top_k_top_p` can keep the whole group, also beyond top-k; the candidate
path keeps the tokens of the group with the lowest ids that top-k and top-p
need.

Gathered layout: `[num_logits, num_shards * 2 * num_candidates]` fp32. Each
shard holds `num_candidates` processed logits (penalties and temperature
applied), then their token ids as fp32.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_logits, tl_rand32

# Largest top-k that can be sampled from candidates: each shard sends this many.
MAX_CANDIDATE_TOP_K = 20

_NO_TOKEN = tl.constexpr(1 << 62)
_NO_CANDIDATE = tl.constexpr(-(1 << 63))
# Candidates kept after the merge: a power of two, at least MAX_CANDIDATE_TOP_K.
_MERGED_CANDIDATES = tl.constexpr(32)


def shard_top_k_supported() -> bool:
    """Whether FlashInfer's top-k with an index tie-break is available."""
    if not has_flashinfer():
        return False
    try:
        from flashinfer.topk import TopKTieBreak  # noqa: F401
    except ImportError:
        return False
    return True


def top_k_shard_candidates(
    logits: torch.Tensor, vocab_start: int, num_candidates: int
) -> torch.Tensor:
    """The `num_candidates` largest processed logits of each row of a
    vocabulary shard (columns from `vocab_start`; ties go to the lower column),
    then their global token ids as fp32: `[num_logits, 2 * num_candidates]`."""
    from flashinfer.topk import TopKTieBreak, top_k

    assert vocab_start + logits.shape[-1] < 2**24, "token ids must be exact in fp32"
    values, columns = top_k(
        logits, num_candidates, deterministic=True, tie_break=TopKTieBreak.SMALL
    )
    return torch.cat([values, (columns + vocab_start).to(values.dtype)], dim=-1)


@triton.jit
def _kept_candidates(
    gathered_ptr,
    gathered_stride,
    logit_idx,
    top_k,
    top_p,
    num_shards,
    num_candidates,
    PADDED_SHARDS: tl.constexpr,
    PADDED_CANDIDATES: tl.constexpr,
):
    """The best candidates of one row over all shards, in the order of
    `apply_top_k_top_p` (by value, ties by the lower token id):
    `(values, token_ids, valid)`. A candidate outside the row's top-k or top-p
    holds -inf."""
    PADDED_SIZE: tl.constexpr = PADDED_SHARDS * PADDED_CANDIDATES
    row_ptr = gathered_ptr + logit_idx * gathered_stride
    shard = tl.arange(0, PADDED_SHARDS)[:, None]
    column = tl.arange(0, PADDED_CANDIDATES)[None, :]
    valid = ((shard < num_shards) & (column < num_candidates)).reshape(PADDED_SIZE)
    offsets = (shard * 2 * num_candidates + column).reshape(PADDED_SIZE)
    values = tl.load(row_ptr + offsets, mask=valid, other=float("-inf"))
    token_ids = tl.load(row_ptr + offsets + num_candidates, mask=valid, other=0.0)

    # One int64 key per candidate that sorts like (value, lower token id
    # first): the fp32 bits made monotonic, then the complement of the id.
    bits = (values + 0.0).to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    keys = (ordered << 32) | (0x7FFFFFFF - token_ids.to(tl.int32)).to(tl.int64)
    keys = tl.topk(tl.where(valid, keys, _NO_CANDIDATE), _MERGED_CANDIDATES)

    valid = keys != _NO_CANDIDATE
    ordered = (keys >> 32).to(tl.int32)
    bits = tl.where(ordered < 0, ordered ^ 0x7FFFFFFF, ordered)
    values = tl.where(valid, bits.to(tl.float32, bitcast=True), float("-inf"))
    token_ids = 0x7FFFFFFF - (keys & 0xFFFFFFFF)

    # Top-p on the top-k tokens: keep the shortest prefix whose mass reaches p.
    rank = tl.arange(0, _MERGED_CANDIDATES)
    in_top_k = valid & (rank < top_k)
    weights = tl.where(in_top_k, tl.exp(values - tl.max(values, axis=0)), 0.0)
    mass_before = tl.cumsum(weights, axis=0) - weights
    in_top_p = (mass_before < top_p * tl.sum(weights, axis=0)) | (rank == 0)
    in_top_p = in_top_p | (top_p >= 1.0)
    values = tl.where(in_top_k & in_top_p, values, float("-inf"))
    return values, token_ids, valid


@triton.jit
def _sample_candidate(
    values, token_ids, valid, seed, pos, temp, USE_FP64: tl.constexpr
):
    noised = gumbel_noised_logits(
        values,
        token_ids,
        valid,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    # Ties go to the lower token id, as in an argmax over the vocabulary.
    is_best = valid & (noised == tl.max(noised, axis=0))
    return tl.min(tl.where(is_best, token_ids, _NO_TOKEN), axis=0)


@triton.jit
def _gumbel_sample_candidates_kernel(
    # [num_logits]
    sampled_ptr,
    # [num_logits, num_shards * 2 * num_candidates]
    gathered_ptr,
    gathered_stride,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    top_k_ptr,
    top_p_ptr,
    temp_ptr,
    seed_ptr,
    # [num_logits]
    pos_ptr,
    num_shards,
    num_candidates,
    PADDED_SHARDS: tl.constexpr,
    PADDED_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    top_k = tl.load(top_k_ptr + req_state_idx, mask=is_valid_req, other=1)
    top_p = tl.load(top_p_ptr + req_state_idx, mask=is_valid_req, other=1.0)
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0)
    seed = tl.load(seed_ptr + req_state_idx, mask=is_valid_req, other=0)
    values, token_ids, valid = _kept_candidates(
        gathered_ptr,
        gathered_stride,
        logit_idx,
        top_k,
        top_p.to(tl.float32),
        num_shards,
        num_candidates,
        PADDED_SHARDS,
        PADDED_CANDIDATES,
    )
    sampled = _sample_candidate(
        values,
        token_ids,
        valid,
        seed,
        tl.load(pos_ptr + logit_idx),
        temp.to(tl.float32),
        USE_FP64,
    )
    tl.store(sampled_ptr + logit_idx, sampled)


def gumbel_sample_candidates(
    gathered: torch.Tensor,  # [num_logits, num_shards * 2 * num_candidates]
    num_shards: int,
    expanded_idx_mapping: torch.Tensor,  # [num_logits]
    top_k: torch.Tensor,  # [max_num_reqs] int32
    top_p: torch.Tensor,  # [max_num_reqs] fp32
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_logits]
    use_fp64: bool = False,
) -> torch.Tensor:
    """Top-k, top-p and `gumbel_sample(..., apply_temperature=False,
    is_drafting=False)` on the gathered candidates of each row. Returns the
    sampled token ids, `[num_logits]`."""
    gathered, num_candidates = _checked_gathered(gathered, num_shards)
    num_logits = gathered.shape[0]
    sampled = gathered.new_empty(num_logits, dtype=torch.int64)
    if num_logits == 0:
        return sampled
    _gumbel_sample_candidates_kernel[(num_logits,)](
        sampled,
        gathered,
        gathered.stride(0),
        expanded_idx_mapping.contiguous(),
        top_k,
        top_p,
        temperature,
        seed,
        pos.contiguous(),
        num_shards,
        num_candidates,
        PADDED_SHARDS=triton.next_power_of_2(num_shards),
        PADDED_CANDIDATES=triton.next_power_of_2(num_candidates),
        USE_FP64=use_fp64,
    )
    return sampled


@triton.jit
def _rejection_candidates_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_logits, num_shards * 2 * num_candidates]
    gathered_ptr,
    gathered_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    top_k_ptr,
    top_p_ptr,
    temp_ptr,
    seed_ptr,
    # [num_logits]
    pos_ptr,
    num_shards,
    num_candidates,
    PADDED_SHARDS: tl.constexpr,
    PADDED_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = end_idx - start_idx - 1
    top_k = tl.load(top_k_ptr + req_state_idx)
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float32)
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    # Accept draft tokens up to the first rejection, as _rejection_kernel does
    # for one-hot drafts.
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 marks a placeholder draft token, which is always rejected.
        is_valid_draft = draft_token >= 0
        if not is_greedy:
            verifying &= is_valid_draft
        if verifying:
            values, token_ids, valid = _kept_candidates(
                gathered_ptr,
                gathered_stride,
                logit_idx,
                top_k,
                top_p,
                num_shards,
                num_candidates,
                PADDED_SHARDS,
                PADDED_CANDIDATES,
            )
            target_argmax = tl.full((), 0, tl.int64)
            if is_greedy:
                is_best = valid & (values == tl.max(values, axis=0))
                target_argmax = tl.min(tl.where(is_best, token_ids, _NO_TOKEN), axis=0)
                accepted = (target_argmax == draft_token) & is_valid_draft
            else:
                # Probability test of a one-hot draft: p(x) > u.
                u = tl_rand32(seed, tl.load(pos_ptr + logit_idx), includes_zero=False)
                is_draft = valid & (token_ids == draft_token)
                target_logit = tl.max(tl.where(is_draft, values, float("-inf")), axis=0)
                maximum = tl.max(values, axis=0)
                normalizer = maximum + tl.log(tl.sum(tl.exp(values - maximum), axis=0))
                accepted = target_logit - normalizer > tl.log(u)
            verifying = accepted
            accepted_length += accepted
            tl.store(
                sampled_ptr + req_idx * sampled_stride + i,
                target_argmax if is_greedy and not accepted else draft_token,
            )

    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))
    resample_idx = start_idx + accepted_length
    is_bonus = resample_idx == end_idx - 1
    if is_greedy and not is_bonus:
        # The target argmax is already in `sampled`.
        return

    # Resample the rejected or bonus token from the target candidates, without
    # the rejected draft token.
    values, token_ids, valid = _kept_candidates(
        gathered_ptr,
        gathered_stride,
        resample_idx,
        top_k,
        top_p,
        num_shards,
        num_candidates,
        PADDED_SHARDS,
        PADDED_CANDIDATES,
    )
    rejected_token = tl.load(
        draft_sampled_ptr + resample_idx + 1, mask=not is_bonus, other=-1
    ).to(tl.int64)
    is_rejected = (token_ids == rejected_token) & (rejected_token >= 0)
    resampled = _sample_candidate(
        tl.where(is_rejected, float("-inf"), values),
        token_ids,
        valid,
        seed,
        tl.load(pos_ptr + resample_idx),
        temp,
        USE_FP64,
    )
    tl.store(sampled_ptr + req_idx * sampled_stride + accepted_length, resampled)


def rejection_sample_candidates(
    gathered: torch.Tensor,  # [num_logits, num_shards * 2 * num_candidates]
    num_shards: int,
    draft_sampled: torch.Tensor,  # [num_logits]
    cu_num_logits: torch.Tensor,  # [num_reqs + 1]
    pos: torch.Tensor,  # [num_logits]
    idx_mapping: torch.Tensor,  # [num_reqs]
    top_k: torch.Tensor,  # [max_num_reqs] int32
    top_p: torch.Tensor,  # [max_num_reqs] fp32
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    num_speculative_steps: int,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k, top-p and `rejection_sample` for one-hot drafts (no draft logits)
    in its standard mode, on the gathered target candidates of each row.

    Returns `sampled` `[num_reqs, num_speculative_steps + 1]` and `num_sampled`
    `[num_reqs]`; only the first `num_sampled` entries of a row are set.

    The accept test sums the candidate probabilities in one reduction, where
    `rejection_sample` sums them per vocabulary block. The two normalizers can
    differ in the last fp32 bit, so a decision can differ when the target
    probability equals the uniform draw to about 1e-7.
    """
    gathered, num_candidates = _checked_gathered(gathered, num_shards)
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    if num_reqs == 0:
        return sampled, num_sampled
    _rejection_candidates_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        gathered,
        gathered.stride(0),
        draft_sampled.contiguous(),
        cu_num_logits,
        idx_mapping,
        top_k,
        top_p,
        temperature,
        seed,
        pos.contiguous(),
        num_shards,
        num_candidates,
        PADDED_SHARDS=triton.next_power_of_2(num_shards),
        PADDED_CANDIDATES=triton.next_power_of_2(num_candidates),
        USE_FP64=use_fp64,
    )
    return sampled, num_sampled


def _checked_gathered(
    gathered: torch.Tensor, num_shards: int
) -> tuple[torch.Tensor, int]:
    assert gathered.ndim == 2 and gathered.dtype == torch.float32
    assert gathered.shape[1] % (2 * num_shards) == 0
    num_candidates = gathered.shape[1] // (2 * num_shards)
    # The merge keeps the best 32 of the padded candidates.
    padded = triton.next_power_of_2(num_shards) * triton.next_power_of_2(num_candidates)
    assert padded >= 32
    return gathered.contiguous(), num_candidates
