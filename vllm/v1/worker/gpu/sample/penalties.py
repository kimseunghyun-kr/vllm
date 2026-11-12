# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Overridden on CPU in vllm/v1/worker/cpu/shm.py.
PENALTIES_BLOCK_SIZE = 8192
# Requests with at most this many output tokens can take the sparse kernel.
MAX_SPARSE_OUTPUT_TOKENS = 4096
SPARSE_BLOCK_SIZE = 1024


class PenaltiesState(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        self.req_states = req_states
        self.device = req_states.device
        max_num_reqs = req_states.max_num_reqs
        vocab_size = req_states.vocab_size

        self.repetition_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.frequency_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.presence_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.use_penalty = np.zeros(max_num_reqs, dtype=bool)

        # Initialize repetition penalty manually because 0 is an invalid value for it.
        self.repetition_penalty.np.fill(1.0)
        self.repetition_penalty.copy_to_uva()

        # Statistics for penalties.
        self.prompt_bin_mask = torch.zeros(
            max_num_reqs, cdiv(vocab_size, 32), dtype=torch.int32, device=self.device
        )
        # TODO(woosuk): This tensor is rarely used but can be very large, taking up
        # GBs of GPU memory. Optimize the memory usage.
        self.output_bin_counts = torch.zeros(
            max_num_reqs, vocab_size, dtype=torch.int32, device=self.device
        )
        # Distinct output tokens of each request, appended by `post_update`.
        self.output_tokens = torch.zeros(
            max_num_reqs,
            MAX_SPARSE_OUTPUT_TOKENS,
            dtype=torch.int32,
            device=self.device,
        )
        self.num_output_tokens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Whether a request's penalties depend on its output tokens only.
        self.use_sparse = np.zeros(max_num_reqs, dtype=bool)

        self._new_penalties_reqs: list[int] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.repetition_penalty.np[req_idx] = sampling_params.repetition_penalty
        self.frequency_penalty.np[req_idx] = sampling_params.frequency_penalty
        self.presence_penalty.np[req_idx] = sampling_params.presence_penalty

        do_penalty = use_penalty(sampling_params)
        self.use_penalty[req_idx] = do_penalty
        self.use_sparse[req_idx] = sampling_params.repetition_penalty == 1.0
        if do_penalty:
            self._new_penalties_reqs.append(req_idx)
        return do_penalty

    def apply_staged_writes(self) -> None:
        if self._new_penalties_reqs:
            idx_mapping = async_tensor_h2d(
                self._new_penalties_reqs, dtype=torch.int64, device=self.device
            )

            prefill_lens = self.req_states.prefill_len.np[self._new_penalties_reqs]
            max_prefill_len = int(prefill_lens.max())
            # A resumed request has output tokens that `bincount` does not list.
            prompt_lens = self.req_states.prompt_len.np[self._new_penalties_reqs]
            self.use_sparse[self._new_penalties_reqs] &= prefill_lens == prompt_lens
            self.num_output_tokens.index_fill_(0, idx_mapping, 0)
            bincount(
                idx_mapping,
                self.req_states.all_token_ids.gpu,
                self.req_states.prompt_len.gpu,
                self.req_states.prefill_len.gpu,
                self.prompt_bin_mask,
                self.output_bin_counts,
                max_prefill_len,
            )
            self._new_penalties_reqs.clear()

        self.repetition_penalty.copy_to_uva()
        self.frequency_penalty.copy_to_uva()
        self.presence_penalty.copy_to_uva()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not np.any(self.use_penalty[ctx.idx_mapping_np]):
            # No request uses penalties. Skip the kernel launch.
            return logits

        max_output_len = self._max_sparse_output_len(ctx)
        if max_output_len is not None:
            apply_sparse_penalties(
                logits,
                ctx.expanded_idx_mapping,
                ctx.input_ids,
                ctx.expanded_local_pos,
                self.frequency_penalty.gpu,
                self.presence_penalty.gpu,
                self.output_tokens,
                self.num_output_tokens,
                self.output_bin_counts,
                max_output_len,
            )
            return logits

        apply_penalties(
            logits,
            ctx.expanded_idx_mapping,
            ctx.input_ids,
            ctx.expanded_local_pos,
            self.repetition_penalty.gpu,
            self.frequency_penalty.gpu,
            self.presence_penalty.gpu,
            self.prompt_bin_mask,
            self.output_bin_counts,
        )
        return logits

    def _max_sparse_output_len(self, ctx: LogitsContext) -> int | None:
        """Upper bound of the batch's output lengths if every row can take the
        sparse kernel, else None."""
        reqs = ctx.idx_mapping_np
        if not np.all(self.use_sparse[reqs] | ~self.use_penalty[reqs]):
            return None
        prompt_lens = self.req_states.prompt_len.np[reqs]
        max_output_len = int(np.max(ctx.seq_lens_upper_bound_np - prompt_lens))
        if max_output_len > MAX_SPARSE_OUTPUT_TOKENS:
            return None
        return max_output_len


@triton.jit
def _sparse_penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    output_tokens_ptr,
    output_tokens_stride,
    num_output_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
    if freq_penalty == 0.0 and pres_penalty == 0.0:
        return

    row_ptr = logits_ptr + token_idx * logits_stride
    counts_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos

    # The request's distinct output tokens, as _penalties_kernel treats them.
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.load(num_output_tokens_ptr + req_state_idx)
    tokens = tl.load(
        output_tokens_ptr + req_state_idx * output_tokens_stride + offsets,
        mask=mask,
        other=0,
    ).to(tl.int64)
    counts = tl.load(counts_ptr + tokens, mask=mask, other=0)
    for prev_pos in tl.range(pos):
        draft = tl.load(token_ids_ptr + start_idx + prev_pos + 1).to(tl.int64)
        counts = counts + (tokens == draft).to(tl.int32)
    logits = tl.load(row_ptr + tokens, mask=mask, other=0.0).to(tl.float32)
    logits -= freq_penalty * counts
    logits -= pres_penalty * (counts > 0)
    tl.store(row_ptr + tokens, logits, mask=mask)

    if block_idx != 0:
        return
    # Draft tokens of earlier positions that are not output tokens yet: each
    # distinct one once, at its first position.
    for prev_pos in tl.range(pos):
        draft = tl.load(token_ids_ptr + start_idx + prev_pos + 1).to(tl.int64)
        if tl.load(counts_ptr + draft) == 0:
            occurrences = 0
            is_first = True
            for other_pos in tl.range(pos):
                other = tl.load(token_ids_ptr + start_idx + other_pos + 1)
                if other.to(tl.int64) == draft:
                    occurrences += 1
                    if other_pos < prev_pos:
                        is_first = False
            if is_first:
                logit = tl.load(row_ptr + draft).to(tl.float32)
                logit -= freq_penalty * occurrences
                logit -= pres_penalty
                tl.store(row_ptr + draft, logit)


def apply_sparse_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    output_tokens: torch.Tensor,
    num_output_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_output_len: int,
) -> None:
    """`apply_penalties` for rows without a repetition penalty, in place and
    bitwise equal: it visits only the tokens a request has produced, not the
    vocabulary. `max_output_len` bounds the requests' output lengths."""
    num_blocks = max(1, triton.cdiv(max_output_len, SPARSE_BLOCK_SIZE))
    _sparse_penalties_kernel[(logits.shape[0], num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        frequency_penalty,
        presence_penalty,
        output_tokens,
        output_tokens.stride(0),
        num_output_tokens,
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=SPARSE_BLOCK_SIZE,
    )


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    # Parenthesize for Triton backends that reject chained `A or B or C`.
    use_penalty = (use_rep_penalty or use_freq_penalty) or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Accumulate draft token counts from previous positions directly into
    # output_bin_counts (preserves its native tensor layout, avoiding an
    # expensive shared-memory layout conversion after the loop).
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos
    output_bin_counts = base_output_counts
    for prev_pos in tl.range(pos):
        prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
        token_match = block == prev_token
        output_bin_counts = output_bin_counts + token_match.to(tl.int32)
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = PENALTIES_BLOCK_SIZE
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _penalties_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _bincount_kernel(
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    prefill_len_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prompt_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        idx = prompt_tokens // 32
        bit_idx = prompt_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx
        tl.atomic_or(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + idx,
            bit,
            mask=mask,
        )

    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        output_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        tl.atomic_add(
            output_bin_counts_ptr
            + req_state_idx * output_bin_counts_stride
            + output_tokens,
            1,
            mask=mask,
        )


def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    # Use index_fill_ instead of `tensor[idx] = 0` to avoid sync.
    idx_long = expanded_idx_mapping.long()
    prompt_bin_mask.index_fill_(0, idx_long, 0)
    output_bin_counts.index_fill_(0, idx_long, 0)
    num_tokens = expanded_idx_mapping.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)
    _bincount_kernel[(num_tokens, num_blocks)](
        expanded_idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )


def use_penalty(sampling_params: SamplingParams) -> bool:
    return (
        sampling_params.repetition_penalty != 1.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
    )
