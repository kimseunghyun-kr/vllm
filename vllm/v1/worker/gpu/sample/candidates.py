# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling and speculative verification on per-row candidate tokens.

With top-k sampling, the sampled token depends only on the k best processed
logits of a row. These functions take those k `(value, token id)` pairs instead
of a `[num_logits, vocab]` matrix. They draw the same uniforms and the same
Gumbel noise as `gumbel_sample` and `rejection_sample`, because both are keyed
by seed, position and token id.

Candidate layout, for every function: `values` and `token_ids` are
`[num_logits, num_candidates]`, each row sorted by ascending token id. `values`
are processed logits in fp32 (temperature, penalties, top-k and top-p already
applied); a masked candidate holds -inf. Tokens outside the candidates are
treated as masked.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32


@triton.jit
def _load_candidates(
    values_ptr,
    token_ids_ptr,
    logit_idx,
    num_candidates,
    PADDED_CANDIDATES: tl.constexpr,
):
    column = tl.arange(0, PADDED_CANDIDATES)
    valid = column < num_candidates
    offsets = logit_idx * num_candidates + column
    values = tl.load(values_ptr + offsets, mask=valid, other=float("-inf"))
    token_ids = tl.load(token_ids_ptr + offsets, mask=valid, other=-1)
    return values.to(tl.float32), token_ids.to(tl.int64), valid


@triton.jit
def _logsumexp(values):
    maximum = tl.max(values, axis=0)
    return maximum + tl.log(tl.sum(tl.exp(values - maximum), axis=0))


@triton.jit
def _gumbel_sample_candidates_kernel(
    sampled_ptr,
    values_ptr,
    token_ids_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    num_candidates,
    PADDED_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    values, token_ids, valid = _load_candidates(
        values_ptr, token_ids_ptr, logit_idx, num_candidates, PADDED_CANDIDATES
    )
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0)
    seed = tl.load(seed_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + logit_idx)
    _, winner = gumbel_noised_argmax(
        values,
        token_ids,
        valid,
        seed,
        pos,
        temp.to(tl.float32),
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    sampled = tl.load(token_ids_ptr + logit_idx * num_candidates + winner)
    tl.store(sampled_ptr + logit_idx, sampled.to(tl.int64))


def gumbel_sample_candidates(
    values: torch.Tensor,  # [num_logits, num_candidates]
    token_ids: torch.Tensor,  # [num_logits, num_candidates]
    expanded_idx_mapping: torch.Tensor,  # [num_logits]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_logits]
    use_fp64: bool = False,
) -> torch.Tensor:
    """`gumbel_sample(..., apply_temperature=False, is_drafting=False)` on the
    candidates of each row. Returns the sampled token ids, `[num_logits]`."""
    values, token_ids = _checked_candidates(values, token_ids)
    num_logits, num_candidates = values.shape
    sampled = token_ids.new_empty(num_logits, dtype=torch.int64)
    if num_logits == 0:
        return sampled
    _gumbel_sample_candidates_kernel[(num_logits,)](
        sampled,
        values,
        token_ids,
        expanded_idx_mapping.contiguous(),
        temperature,
        seed,
        pos.contiguous(),
        num_candidates,
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
    # [num_logits, num_candidates]
    values_ptr,
    token_ids_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    seed_ptr,
    # [num_logits]
    pos_ptr,
    num_candidates,
    PADDED_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = end_idx - start_idx - 1
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
            values, token_ids, valid = _load_candidates(
                values_ptr, token_ids_ptr, logit_idx, num_candidates, PADDED_CANDIDATES
            )
            target_argmax = tl.full((), 0, tl.int64)
            if is_greedy:
                _, argmax_column = tl.max(values, axis=0, return_indices=True)
                target_argmax = tl.load(
                    token_ids_ptr + logit_idx * num_candidates + argmax_column
                ).to(tl.int64)
                accepted = (target_argmax == draft_token) & is_valid_draft
            else:
                # Probability test of a one-hot draft: p(x) > u.
                u = tl_rand32(seed, tl.load(pos_ptr + logit_idx), includes_zero=False)
                is_draft = valid & (token_ids == draft_token)
                target_logit = tl.max(tl.where(is_draft, values, float("-inf")), axis=0)
                accepted = target_logit - _logsumexp(values) > tl.log(u)
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
    values, token_ids, valid = _load_candidates(
        values_ptr, token_ids_ptr, resample_idx, num_candidates, PADDED_CANDIDATES
    )
    rejected_token = tl.load(
        draft_sampled_ptr + resample_idx + 1, mask=not is_bonus, other=-1
    ).to(tl.int64)
    is_rejected = (token_ids == rejected_token) & (rejected_token >= 0)
    values = tl.where(is_rejected, float("-inf"), values)
    _, winner = gumbel_noised_argmax(
        values,
        token_ids,
        valid,
        seed,
        tl.load(pos_ptr + resample_idx),
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    resampled = tl.load(token_ids_ptr + resample_idx * num_candidates + winner)
    tl.store(
        sampled_ptr + req_idx * sampled_stride + accepted_length,
        resampled.to(tl.int64),
    )


def rejection_sample_candidates(
    values: torch.Tensor,  # [num_logits, num_candidates]
    token_ids: torch.Tensor,  # [num_logits, num_candidates]
    draft_sampled: torch.Tensor,  # [num_logits]
    cu_num_logits: torch.Tensor,  # [num_reqs + 1]
    pos: torch.Tensor,  # [num_logits]
    idx_mapping: torch.Tensor,  # [num_reqs]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    num_speculative_steps: int,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`rejection_sample` for one-hot drafts (no draft logits) in its standard
    mode, on the target candidates of each row.

    Returns `sampled` `[num_reqs, num_speculative_steps + 1]` and `num_sampled`
    `[num_reqs]`; only the first `num_sampled` entries of a row are set.

    The accept test sums the candidate probabilities in one reduction, where
    `rejection_sample` sums them per vocabulary block. The two normalizers can
    differ in the last fp32 bit, so a decision can differ when the target
    probability equals the uniform draw to about 1e-7.
    """
    values, token_ids = _checked_candidates(values, token_ids)
    num_candidates = values.shape[1]
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
        values,
        token_ids,
        draft_sampled.contiguous(),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos.contiguous(),
        num_candidates,
        PADDED_CANDIDATES=triton.next_power_of_2(num_candidates),
        USE_FP64=use_fp64,
        num_warps=1,
    )
    return sampled, num_sampled


def top_k_shard_candidates(
    logits: torch.Tensor, vocab_start: int, num_candidates: int
) -> torch.Tensor:
    """The `num_candidates` largest processed logits of each row of a
    vocabulary shard (columns from `vocab_start`), then their global token ids
    as fp32: `[num_logits, 2 * num_candidates]`, ready for one all-gather."""
    assert vocab_start + logits.shape[-1] < 2**24, "token ids must be exact in fp32"
    values, columns = torch.topk(logits, num_candidates, dim=-1)
    return torch.cat([values, (columns + vocab_start).to(values.dtype)], dim=-1)


def merge_shard_candidates(
    gathered: torch.Tensor, num_shards: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the last-dim all-gather of `top_k_shard_candidates` to the
    `num_candidates` best `(values, token_ids)` of the whole vocabulary."""
    num_logits = gathered.shape[0]
    num_candidates = gathered.shape[1] // (2 * num_shards)
    gathered = gathered.view(num_logits, num_shards, 2, num_candidates)
    values, chosen = torch.topk(
        gathered[:, :, 0].reshape(num_logits, -1), num_candidates, dim=-1
    )
    token_ids = gathered[:, :, 1].reshape(num_logits, -1).gather(1, chosen)
    return values, token_ids.to(torch.int32)


@triton.jit
def _finalize_candidates_kernel(
    out_values_ptr,
    out_token_ids_ptr,
    values_ptr,
    token_ids_ptr,
    expanded_idx_mapping_ptr,
    top_k_ptr,
    top_p_ptr,
    num_candidates,
    PADDED_CANDIDATES: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    values, token_ids, valid = _load_candidates(
        values_ptr, token_ids_ptr, logit_idx, num_candidates, PADDED_CANDIDATES
    )
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    top_k = tl.load(top_k_ptr + req_state_idx)
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float32)

    # Order of the stock top-k/top-p: by value, ties by the lower token id.
    other_values = values[None, :]
    other_ids = token_ids[None, :]
    precedes = (other_values > values[:, None]) | (
        (other_values == values[:, None]) & (other_ids < token_ids[:, None])
    )
    precedes = precedes & valid[None, :]
    rank = tl.sum(precedes.to(tl.int32), axis=1)
    in_top_k = valid & (rank < top_k)

    # Top-p on the top-k tokens: keep the shortest prefix whose mass reaches p.
    maximum = tl.max(tl.where(in_top_k, values, float("-inf")), axis=0)
    weight = tl.where(in_top_k, tl.exp(values - maximum), 0.0)
    mass_before = tl.sum(tl.where(precedes, weight[None, :], 0.0), axis=1)
    in_top_p = (mass_before < top_p * tl.sum(weight, axis=0)) | (rank == 0)
    kept = in_top_k & (in_top_p | (top_p >= 1.0))

    # Write the row in ascending token id order.
    column = tl.sum(
        (valid[None, :] & (other_ids < token_ids[:, None])).to(tl.int32), axis=1
    )
    offsets = logit_idx * num_candidates + column
    tl.store(
        out_values_ptr + offsets, tl.where(kept, values, float("-inf")), mask=valid
    )
    tl.store(out_token_ids_ptr + offsets, token_ids, mask=valid)


def finalize_candidates(
    values: torch.Tensor,  # [num_logits, num_candidates], any order
    token_ids: torch.Tensor,  # [num_logits, num_candidates], distinct per row
    expanded_idx_mapping: torch.Tensor,  # [num_logits]
    top_k: torch.Tensor,  # [max_num_reqs] int32
    top_p: torch.Tensor,  # [max_num_reqs] fp32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply top-k, then top-p, to each row's candidates and sort the row by
    token id: the candidate layout of this module.

    Ties are ordered by the lower token id, as in `apply_top_k_top_p`. A row
    must hold more candidates than its top-k tokens tied at the k-th value.
    """
    values, token_ids = _checked_candidates(values, token_ids)
    num_logits, num_candidates = values.shape
    out_values = torch.empty_like(values)
    out_token_ids = torch.empty_like(token_ids, dtype=torch.int64)
    if num_logits == 0:
        return out_values, out_token_ids
    _finalize_candidates_kernel[(num_logits,)](
        out_values,
        out_token_ids,
        values,
        token_ids,
        expanded_idx_mapping.contiguous(),
        top_k,
        top_p,
        num_candidates,
        PADDED_CANDIDATES=triton.next_power_of_2(num_candidates),
    )
    return out_values, out_token_ids


def _checked_candidates(
    values: torch.Tensor, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert values.ndim == 2 and values.shape == token_ids.shape
    assert values.dtype == torch.float32
    assert token_ids.dtype in (torch.int32, torch.int64)
    return values.contiguous(), token_ids.contiguous()
