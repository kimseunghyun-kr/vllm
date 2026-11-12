# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState, apply_penalties
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="the penalties kernels need a GPU"
)

VOCAB_SIZE = 4096
PROMPT_LEN = 8
NUM_STEPS = 3  # speculative steps: each request has NUM_STEPS + 1 logits rows


@pytest.mark.parametrize(
    "sampling_params",
    [
        [SamplingParams(presence_penalty=1.5)],
        [SamplingParams(frequency_penalty=0.2, presence_penalty=-0.5)],
        [SamplingParams(presence_penalty=1.5), SamplingParams()],
        [SamplingParams(presence_penalty=1.5), SamplingParams(repetition_penalty=1.1)],
    ],
    ids=["presence", "frequency+presence", "one-without-penalty", "one-repetition"],
)
def test_penalties_match_dense_kernel_over_decode_steps(sampling_params):
    """`PenaltiesState.apply` must give the logits of the dense kernel at every
    step of a decode with draft tokens, also where it takes the sparse kernel
    (no repetition penalty in the batch)."""
    device = torch.device("cuda")
    num_reqs = len(sampling_params)
    req_states = RequestState(
        max_num_reqs=num_reqs,
        max_model_len=256,
        max_num_batched_tokens=64,
        num_speculative_steps=NUM_STEPS,
        vocab_size=VOCAB_SIZE,
        device=device,
    )
    penalties = PenaltiesState(
        VllmConfig(), LogitsProcRequestState.from_request_state(req_states)
    )
    generator = torch.Generator(device=device).manual_seed(0)
    for req_idx in range(num_reqs):
        prompt = torch.randint(
            0, VOCAB_SIZE, (PROMPT_LEN,), generator=generator, device=device
        )
        req_states.add_request(str(req_idx), PROMPT_LEN, prompt.tolist(), 0, 128)
    slots = [req_states.req_id_to_index[str(i)] for i in range(num_reqs)]
    for slot, params in zip(slots, sampling_params):
        penalties.add_request(slot, params)
    req_states.apply_staged_writes()
    penalties.apply_staged_writes()

    idx_mapping_np = np.array(slots, dtype=np.int32)
    idx_mapping = torch.from_numpy(idx_mapping_np).to(device)
    expanded = idx_mapping.repeat_interleave(NUM_STEPS + 1).contiguous()
    local_pos = torch.arange(NUM_STEPS + 1, dtype=torch.int32, device=device)
    local_pos = local_pos.repeat(num_reqs)
    num_rows = expanded.shape[0]
    num_outputs = 0
    for _ in range(12):
        logits = torch.randn(num_rows, VOCAB_SIZE, generator=generator, device=device)
        # Tokens repeat within a step and across steps (64 possible values).
        tokens = torch.randint(0, 64, (num_rows,), generator=generator, device=device)
        ctx = LogitsContext(
            expanded_idx_mapping=expanded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_local_pos=local_pos,
            input_ids=tokens,
            pos=torch.zeros(num_rows, dtype=torch.int64, device=device),
            # Padded with zeros, as the runner pads it to its CUDA graph size.
            seq_lens_upper_bound_np=np.pad(
                np.full(
                    num_reqs, PROMPT_LEN + num_outputs + NUM_STEPS + 1, dtype=np.int32
                ),
                (0, 3),
            ),
        )
        expected = logits.clone()
        apply_penalties(
            expected,
            expanded,
            tokens,
            local_pos,
            penalties.repetition_penalty.gpu,
            penalties.frequency_penalty.gpu,
            penalties.presence_penalty.gpu,
            penalties.prompt_bin_mask,
            penalties.output_bin_counts,
        )
        assert torch.equal(penalties.apply(logits.clone(), ctx), expected)

        # Commit the first two tokens of every request's rows.
        sampled = tokens.view(num_reqs, NUM_STEPS + 1)[:, :2].contiguous()
        num_sampled = torch.full((num_reqs,), 2, dtype=torch.int32, device=device)
        post_update(
            idx_mapping,
            req_states.num_computed_tokens.gpu,
            req_states.last_sampled_tokens,
            penalties.output_bin_counts,
            sampled,
            num_sampled,
            torch.zeros_like(num_sampled),
            None,
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
            output_tokens=penalties.output_tokens,
            num_output_tokens=penalties.num_output_tokens,
        )
        num_outputs += 2
    # The sparse kernel's list holds each request's distinct output tokens.
    assert torch.equal(
        penalties.num_output_tokens[idx_mapping.long()],
        (penalties.output_bin_counts[idx_mapping.long()] > 0).sum(dim=-1).int(),
    )
