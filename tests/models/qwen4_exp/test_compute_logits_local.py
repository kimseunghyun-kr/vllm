# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp exposes vocab-shard logits for batch-sharded sampling."""

import torch
from torch import nn

# NVIDIA only: the NVIDIA and AMD model modules register the same custom ops.
from vllm.models.qwen4_exp.nvidia import model


def _causal_lm_with_recording_processor(model_cls):
    calls: list[tuple[nn.Module, torch.Tensor, bool]] = []

    def logits_processor(
        lm_head: nn.Module, hidden_states: torch.Tensor, *, skip_gather: bool = False
    ) -> torch.Tensor:
        calls.append((lm_head, hidden_states, skip_gather))
        return hidden_states + 1

    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.lm_head = nn.Identity()
    model.logits_processor = logits_processor
    return model, calls


def test_compute_logits_local_skips_gather() -> None:
    causal_lm, calls = _causal_lm_with_recording_processor(model.Qwen4ExpForCausalLM)
    hidden_states = torch.tensor([4.0])

    result = causal_lm.compute_logits_local(hidden_states)

    assert torch.equal(result, torch.tensor([5.0]))
    assert calls == [(causal_lm.lm_head, hidden_states, True)]


def test_vl_wrapper_delegates_compute_logits_local() -> None:
    language_model, calls = _causal_lm_with_recording_processor(
        model.Qwen4ExpForCausalLM
    )
    wrapper = object.__new__(model.Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = language_model
    hidden_states = torch.tensor([1.0])

    result = wrapper.compute_logits_local(hidden_states)

    assert torch.equal(result, torch.tensor([2.0]))
    assert calls == [(language_model.lm_head, hidden_states, True)]
