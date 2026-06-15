# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""EagerRunner — the no-cuda-graph phase runner.

The eager dual of the cuda-graph runners. Where ``DecodeCudaGraphRunner`` and
``PrefillCudaGraphRunner`` capture a ``torch.cuda.CUDAGraph`` per shape and
replay it, ``EagerRunner`` runs ``model.forward`` live each iteration over one
fixed-max static buffer set. It is used when CUDA graph is disabled for the
generation phases (``--disable-cuda-graph`` / a phase resolving to
``disabled``).

When CUDA graph is disabled, ``ModelRunner.decode_cuda_graph_runner`` and
``prefill_cuda_graph_runner`` point at ONE ``EagerRunner`` instance,
mode-dispatched on ``forward_batch.forward_mode`` (decode / extend / idle).

This is a real extraction — the eager path that used to live inline in
``ModelRunner.forward_decode`` / ``forward_extend`` (eager branch) /
``forward_idle`` (plus ``_eager_fb_view`` and the eager input registry) now
lives here. The cuda-graph runners stay purely cuda-graph (no ``eager`` flag, no
``EagerBackend``). ``load_batch`` copies the live batch into the eager static
buffers (``_eager_fb_view``); ``execute`` inits attention metadata and runs
``model.forward`` live, mode-dispatched.

This runner allocates ONE fixed-max static buffer set in ``__init__`` (sized once
at the prefill token ceiling, no grow); ``load_batch`` returns the registry's
``extract_buffer`` view sliced to the batch.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Tuple, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_eager_registry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner import ModelRunner


class EagerRunner(BaseRunner):
    """No-cuda-graph phase runner; mode-dispatched over decode + extend + idle.

    Public surface (the :class:`BaseRunner` ABC):
      - can_run_graph(forward_batch) -> False (always; the dispatch gate that
        keeps callers from routing an eager batch into a graph-replay branch).
      - warmup() — inherited; run-once kernel warmup + flashinfer autotune, run
        in __init__ (eager has no capture step), before any forward.
      - load_batch(forward_batch, ...) — copy the live batch into the eager
        static buffers (the one fixed-max registry, sliced to the batch).
      - execute(forward_batch, ...) — init attention metadata + run
        model.forward live, mode-dispatched (decode / extend / idle).
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        mr = model_runner
        sa = mr.server_args
        # One fixed-max static buffer set, allocated ONCE (no grow): the eager
        # runner is built before the cuda-graph runners, so its (largest) buffers
        # are canonical in the shared input pool and the cg runners coalesce onto
        # them. Sized to the true ceilings — max_bs running requests and the
        # prefill token ceiling (a decode batch is a [:num_tokens] prefix).
        # tokens-per-bs mirrors the cuda-graph runners so the fixed-max buffers
        # cover every batch that can fall back to eager. The target/main worker
        # uses TARGET_VERIFY tokens; the draft worker runs several modes with
        # different tokens/req — draft decode = topk, draft extend =
        # speculative_num_draft_tokens, multi-layer draft extend =
        # speculative_num_steps + 1 + step (< 2*speculative_num_steps) — so size
        # for the max; dLLM uses block_size.
        num_tokens_per_bs = 1
        if mr.spec_algorithm.is_speculative():
            if mr.is_draft_worker:
                num_tokens_per_bs = max(
                    sa.speculative_eagle_topk or 1,
                    sa.speculative_num_draft_tokens or 1,
                    2 * (sa.speculative_num_steps or 0),
                )
            else:
                num_tokens_per_bs = (
                    mr.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                        sa.speculative_num_draft_tokens, mr.is_draft_worker
                    )
                )
        else:
            dllm_config = DllmConfig.from_server_args(sa)
            if dllm_config is not None:
                # dLLM runs block_size tokens/request (DLLM_EXTEND).
                num_tokens_per_bs = dllm_config.block_size
        max_bs = mr.max_running_requests
        if (
            mr.is_draft_worker
            and mr.spec_algorithm.is_frozen_kv_mtp()
            and sa.speculative_eagle_topk > 1
        ):
            # Frozen-KV MTP expands the draft batch by topk on the bs axis
            # (expand_for_topk_draft) before the eager fallback.
            max_bs *= sa.speculative_eagle_topk
        prefill_ceiling = (
            sa.chunked_prefill_size
            if sa.chunked_prefill_size and sa.chunked_prefill_size > 0
            else mr.max_total_num_tokens
        )
        max_num_token = max(prefill_ceiling, max_bs * num_tokens_per_bs)
        is_encoder_decoder = mr.model_config.is_encoder_decoder
        self._eager_registry = build_eager_registry(
            device=mr.device,
            max_bs=max_bs,
            max_num_token=max_num_token,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=(
                sa.enable_mamba_extra_buffer() and mr.spec_algorithm.is_none()
            ),
            is_encoder_decoder=is_encoder_decoder,
            encoder_len_fill_value=(
                getattr(mr.model_config.hf_config, "max_source_positions", 0)
                if is_encoder_decoder
                else 0
            ),
            dp_size=sa.dp_size,
        )
        # Eager has no capture step, so it warms up kernels here in __init__
        # (run-once across all runners via ModelRunner._kernel_warmed_up; a cheap
        # no-op if a cuda-graph runner already warmed up). Built before the cg
        # runners, so this autotune precedes their capture.
        self.warmup()

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        # Eager never runs a cuda graph; callers dispatch on isinstance(...,
        # EagerRunner) and must not route an eager batch into a replay branch.
        return False

    def load_batch(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> ForwardBatch:
        """Copy the live batch into the fixed-max eager static buffers (sliced to
        this batch's shape) — the eager counterpart of the cuda-graph runners'
        load_batch."""
        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
            return replace(forward_batch)
        raw_bs = forward_batch.batch_size
        raw_num_tokens = forward_batch.input_ids.shape[0]
        registry = self._eager_registry
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=raw_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=raw_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return registry.extract_buffer(
            padded_bs=raw_bs,
            padded_num_tokens=raw_num_tokens,
            forward_batch_template=forward_batch,
        )

    def execute(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> Any:
        mode = forward_batch.forward_mode
        if mode.is_decode():
            return self._execute_decode(forward_batch, pp_proxy_tensors)
        if mode.is_idle():
            return self._execute_idle(forward_batch, pp_proxy_tensors)
        if mode.is_extend(include_draft_extend_v2=True):
            return self._execute_extend(forward_batch, pp_proxy_tensors)
        raise ValueError(f"Invalid forward mode for eager runner: {mode}")

    def _resolve_decode_pdmux(
        self,
    ) -> Tuple[Any, contextlib.AbstractContextManager]:
        """Resolve the (attn_backend, forward_context) the eager decode forward
        runs under. PDmux selects a per-stream backend and publishes it via an
        active ForwardContext; non-pdmux uses attn_backend + the ambient ctx."""
        model_runner = self.model_runner
        if model_runner.server_args.enable_pdmux:
            return model_runner.decode_attn_backend, forward_context(
                ForwardContext(attn_backend=model_runner.decode_attn_backend)
            )
        return model_runner.attn_backend, contextlib.nullcontext()

    def _execute_decode(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        enable_pdmux = model_runner.server_args.enable_pdmux
        attn_backend, pdmux_ctx = self._resolve_decode_pdmux()
        if not enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)

        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "decode"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )

        with ctx, pdmux_ctx:
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    def _execute_extend(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        model_runner = self.model_runner
        kwargs = model_runner._extend_forward_kwargs(forward_batch, pp_proxy_tensors)

        if not model_runner.server_args.enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)

        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            model_runner.attn_backend.init_forward_metadata(forward_batch)

        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "extend"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            pcg_runner = model_runner.prefill_cuda_graph_runner
            if (
                _is_hip
                and pcg_runner is not None
                and not isinstance(pcg_runner, EagerRunner)
            ):
                # AMD/HIP: when PCG is enabled but the batch exceeds max captured
                # size, run eagerly under enable_tc_piecewise_cuda_graph() and
                # set_tc_piecewise_forward_context() so that (a) Dynamo guards on
                # _in_tc_piecewise_cuda_graph stay consistent with the PCG-traced
                # graph (preventing runtime recompilation) and (b) PCG-specific
                # code paths (MoE, attention) can access their layer objects.
                with (
                    enable_tc_piecewise_cuda_graph(),
                    set_tc_piecewise_forward_context(
                        forward_batch,
                        model_runner.attention_layers,
                        getattr(model_runner.model, "quant_config", None),
                        model_runner.moe_layers,
                        model_runner.moe_fusions,
                        dsa_indexers=model_runner.dsa_indexers,
                    ),
                ):
                    ret = model_runner.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
            else:
                ret = model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
        return ret

    def _execute_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        # In DP Attention, IDLE batches may be padded (batch_size > 0) for MLP
        # sync. Reinit metadata for the padded case so attention kernels see
        # the right batch_size (e.g. DSA Indexer). For the unpadded case
        # (batch_size == 0) explicitly drop any stale forward_metadata left
        # over from the previous forward — without this, attention layers
        # called from the idle path can re-read a prior batch's req_pool
        # indices and trigger SWA mapping use-after-free.
        if forward_batch.batch_size > 0:
            if not model_runner.server_args.enable_pdmux:
                forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
        else:
            model_runner.attn_backend.forward_metadata = None

        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)
        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "idle"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
