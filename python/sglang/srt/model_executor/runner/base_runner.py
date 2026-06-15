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
"""BaseRunner — the surface shared by every phase runner.

Two kinds of runner subclass this:

  - ``BaseCudaGraphRunner`` (and its ``Decode``/``PrefillCudaGraphRunner``
    subclasses) — capture a ``torch.cuda.CUDAGraph`` per shape and replay it.
  - ``EagerRunner`` (upcoming) — no capture; runs ``model.forward`` live each
    iteration over one static batch.

``BaseRunner`` holds only what both share: the shared ``__init__``, the run-once
kernel ``warmup()`` (flashinfer autotune + PP-DeepGEMM warmup) plus the
autotune / dummy-run machinery it drives, and the abstract per-iteration entry
points (``can_run_graph`` dispatch gate, ``load_batch`` copy-into-static,
``execute`` run). All capture/shape machinery (``capture`` / ``capture_prepare``
/ ``capture_one_shape`` / bucket padding / the graph-only ``ExecutionBackend``)
lives on ``BaseCudaGraphRunner``.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    NgramEmbeddingInfo,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.spec_info import create_dummy_verify_input
from sglang.srt.utils import (
    empty_context,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
)
from sglang.srt.utils.common import ceil_align, require_mlp_sync

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _allocate_decode_buffers(
    *,
    device: torch.device,
    max_bs: int,
    max_num_token: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    dp_size: int,
    pp_size: int,
    is_encoder_decoder: bool,
    require_mlp_tp_gather: bool,
    seq_len_fill_value: int,
    encoder_len_fill_value: int,
    num_tokens_per_bs: int,
    cache_loc_dtype: torch.dtype,
    enable_mamba_track: bool,
    ne_token_table: Optional[torch.Tensor] = None,
    hc_hidden_size: Optional[int] = None,
) -> SimpleNamespace:
    """Allocate the FB-shared decode buffers as a namespace adopted by
    ``build_decode_registry(source=...)``."""
    with torch.device(device):
        input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
        input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
        req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
        seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int64)
        out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
        positions = torch.zeros((max_num_token,), dtype=torch.int64)
        mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
        num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
        custom_mask = torch.ones(
            (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_bs,
            dtype=torch.bool,
        )
        next_token_logits_buffer = torch.zeros(
            (max_num_token, vocab_size),
            dtype=torch.float,
        )
        mamba_track_indices = (
            torch.zeros((max_bs,), dtype=torch.int64) if enable_mamba_track else None
        )
        mamba_track_mask = (
            torch.zeros((max_bs,), dtype=torch.bool) if enable_mamba_track else None
        )

        if pp_size > 1:
            # mHC (e.g. DSV4) flattens residual into hidden_states (size = hc_hidden_size).
            is_mhc = hc_hidden_size is not None
            hs = hc_hidden_size if is_mhc else hidden_size
            pp_proxy_tensors = {
                "hidden_states": torch.zeros((max_bs, hs), dtype=dtype),
            }
            if not is_mhc:
                pp_proxy_tensors["residual"] = torch.zeros(
                    (max_bs, hidden_size), dtype=dtype
                )
        else:
            pp_proxy_tensors = None

        if is_encoder_decoder:
            encoder_lens = torch.full(
                (max_bs,), encoder_len_fill_value, dtype=torch.int32
            )
        else:
            encoder_lens = None

        if require_mlp_tp_gather:
            global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
            global_num_tokens_for_logprob_gpu = torch.zeros(
                (dp_size,), dtype=torch.int32
            )
        else:
            global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
            global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)

        ngram_embedding_info = (
            NgramEmbeddingInfo(
                token_table=ne_token_table,
                column_starts=torch.zeros([max_bs], dtype=torch.int32),
                req_lens=torch.ones([max_bs], dtype=torch.int32),
                out_column_starts=torch.zeros([max_bs], dtype=torch.int32),
                out_req_lens=torch.ones([max_bs], dtype=torch.int32),
            )
            if ne_token_table is not None
            else None
        )

        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            rids_int = torch.zeros((max_bs,), dtype=torch.int64)
            bootstrap_room_ids_int = torch.full((max_bs,), -1, dtype=torch.int64)
        else:
            rids_int = None
            bootstrap_room_ids_int = None

    seq_lens_cpu = torch.full(
        (max_bs,),
        seq_len_fill_value,
        dtype=torch.int64,
        device="cpu",
    )

    return SimpleNamespace(
        input_ids=input_ids,
        input_embeds=input_embeds,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        out_cache_loc=out_cache_loc,
        positions=positions,
        mrope_positions=mrope_positions,
        num_token_non_padded=num_token_non_padded,
        custom_mask=custom_mask,
        next_token_logits_buffer=next_token_logits_buffer,
        mamba_track_indices=mamba_track_indices,
        mamba_track_mask=mamba_track_mask,
        encoder_lens=encoder_lens,
        global_num_tokens_gpu=global_num_tokens_gpu,
        global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        pp_proxy_tensors=pp_proxy_tensors,
        ngram_embedding_info=ngram_embedding_info,
        rids_int=rids_int,
        bootstrap_room_ids_int=bootstrap_room_ids_int,
    )


class BaseRunner(ABC):
    """Abstract base shared by the cuda-graph runners and the eager runner.

    Methods:
      - can_run_graph(forward_batch) — should forward_batch go through cuda
        graph replay (vs eager fallback)? Dispatch gate; the eager runner
        always returns True.
      - warmup() — one-time kernel warmup / flashinfer autotune, run once
        across whichever runners exist (decode + prefill cuda-graph runners,
        or the eager runner).
      - load_batch(forward_batch, ...) — copy the live fb into the runner's
        static buffers and refresh dynamic attention metadata.
      - execute(forward_batch, ...) — run one batch (graph replay for the
        cuda-graph runners; model.forward for eager) and slice to raw size.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tbo_plugin = TboCudaGraphRunnerPlugin()

    def warmup(self) -> None:
        """Warm up + autotune kernels once — part of the Runner lifecycle.

        Shared by every runner: the cuda-graph runners call this from capture()
        (so the autotune runs right before the graph it tunes), and the eager
        runner calls it on its first execute() (no capture step). Run-once
        across whichever runners exist via the _kernel_warmed_up flag on the
        shared ModelRunner: whoever runs first does the warmup; the rest no-op.
        The autotune / DeepGEMM dummy-forward machinery (_dummy_run /
        _flashinfer_autotune and friends) lives on BaseRunner now and reads the
        shared ModelRunner state through self.model_runner; warmup() is the
        orchestration that drives it.

        The flashinfer-autotune dummy forward reuses this runner's already-
        allocated static decode buffers (via _autotune_buffers) instead of
        allocating a throwaway set; runners with no reusable decode buffers (the
        prefill cuda-graph runner, the eager runner) return (None, None) and we
        fall back to a freshly-allocated dummy set sized to req_to_token_pool.size.
        """
        mr = self.model_runner
        if getattr(mr, "_kernel_warmed_up", False):
            return
        mr._kernel_warmed_up = True

        if mr.device != "cuda":
            return

        # Pre-initialize flashinfer allreduce fusion workspaces before any
        # autotune / capture, so collective ops do not run inside a graph
        # capture context (see _pre_initialize_flashinfer_allreduce_workspace).
        self._pre_initialize_flashinfer_allreduce_workspace()

        if self._should_run_flashinfer_autotune():
            buffers, batch_size = self._autotune_buffers()
            if buffers is None:
                # No reusable static buffers, so build a dummy decode set sized
                # to req_to_token_pool.size. Spec target workers verify
                # num_tokens_per_bs tokens/req, so size for that.
                batch_size = mr.req_to_token_pool.size
                num_tokens_per_bs = 1
                if mr.spec_algorithm.is_speculative() and not mr.is_draft_worker:
                    num_tokens_per_bs = (
                        mr.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                            mr.server_args.speculative_num_draft_tokens,
                            mr.is_draft_worker,
                        )
                    )
                # MLP sync pads token counts to a multiple of attn_tp_size; the
                # dummy forward must match or its collectives mismatch.
                if require_mlp_sync(mr.server_args) and self.attn_tp_size > 1:
                    batch_size = ceil_align(batch_size, self.attn_tp_size)
                buffers = self._alloc_dummy_decode_buffers(
                    batch_size, num_tokens_per_bs=num_tokens_per_bs
                )
            self._flashinfer_autotune(buffers=buffers, batch_size=batch_size)

        if (
            envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.get()
            and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and mr.pp_size > 1
            and not mr.spec_algorithm.is_speculative()
        ):
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                pp_parallel_deep_gemm_warmup,
            )

            pp_parallel_deep_gemm_warmup(self)

    def _pre_initialize_flashinfer_allreduce_workspace(self):
        """Pre-initialize flashinfer allreduce fusion workspaces.

        Must run before CUDA graph capture to avoid collective operations
        (broadcasts, barriers) inside the graph capture context, which can
        deadlock with custom_all_reduce.register_graph_buffers.
        """
        mr = self.model_runner
        if mr.server_args.flashinfer_allreduce_fusion_backend is None:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import pre_initialize_workspaces

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=mr.model_config.hidden_size,
            dtype=mr.dtype,
        )

    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run."""
        mr = self.model_runner
        if mr.server_args.disable_flashinfer_autotune:
            return False

        # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
        # be autotuned -- its _dummy_run would dispatch more tokens per rank than
        # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
        # Read server_args directly to avoid depending on initialize_moe_config()
        # having already populated the MoE backend globals.
        if (
            mr.server_args.moe_runner_backend == "flashinfer_cutedsl"
            and mr.server_args.moe_a2a_backend == "deepep"
        ):
            return False

        backend_str = mr.server_args.moe_runner_backend

        # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

        moe_needs_autotune = backend_str in [
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
            "flashinfer_mxfp4",
            "flashinfer_cutedsl",
            "flashinfer_cutlass",
        ]

        from sglang.srt.layers.quantization.fp4_utils import (
            get_fp4_gemm_runner_backend,
        )

        model_uses_fp4 = mr.model_config.quantization in (
            "modelopt_fp4",
            "modelopt_mixed",
        )
        fp4_gemm_needs_autotune = model_uses_fp4 and (
            get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
            or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
        )

        if not (moe_needs_autotune or fp4_gemm_needs_autotune):
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False

        if mr.spec_algorithm.is_speculative():
            return not mr.is_draft_worker

        return True

    def _flashinfer_autotune(self, *, buffers, batch_size):
        """Run flashinfer autotune.

        buffers / batch_size: a prepared static decode-buffer set and its bs,
        reused for the dummy forward instead of allocating a throwaway set.
        Supplied by warmup() (the decode runner's captured buffers when a graph
        runner exists; a freshly-allocated dummy set in the eager path).
        """
        from flashinfer.autotuner import autotune

        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        mr = self.model_runner
        cache_path = self._flashinfer_autotune_cache_path()
        if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
            autotune_cache = cache_path
            logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = cache_path.parent / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            autotune_cache = (
                runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
            )
            logger.info(
                "Running FlashInfer autotune (cache reuse DISABLED via "
                "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
                autotune_cache,
            )

        # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
        # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
        mr.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.get_device_module(mr.device).stream(mr.forward_stream):
            with (
                torch.inference_mode(),
                autotune(True, cache=str(autotune_cache)),
                autotune_dummy_run_mode(),
            ):
                self._dummy_run(batch_size=batch_size, buffers=buffers)
        torch.cuda.current_stream().wait_stream(mr.forward_stream)
        logger.info("FlashInfer autotune completed.")

    def _flashinfer_autotune_cache_path(self) -> Path:
        import flashinfer

        mr = self.model_runner
        major, minor = torch.cuda.get_device_capability(mr.device)
        arch = f"sm{major}{minor}"
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

        server_args = mr.server_args
        model_key = "|".join(
            [
                str(server_args.model_path),
                str(mr.dtype),
                str(server_args.quantization),
                str(server_args.moe_runner_backend),
                str(mr.tp_size),
                str(mr.pp_size),
                str(mr.dp_size),
                str(mr.moe_ep_size),
                str(mr.model_config.hf_config.__class__.__name__),
            ]
        )
        cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
        cache_dir = (
            Path(envs.SGLANG_CACHE_DIR.get())
            / "flashinfer"
            / "autotune"
            / flashinfer_version
            / arch
            / cache_key
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir / f"rank_tp{mr.tp_rank}_pp{mr.pp_rank}_dp{mr.dp_rank or 0}.json"
        )

    def _alloc_dummy_decode_buffers(self, max_bs: int, *, num_tokens_per_bs: int = 1):
        """Allocate one static decode-buffer set for a dummy forward, sized to
        (max_bs, max_bs * num_tokens_per_bs).

        For callers that have no pre-allocated static buffers to reuse -- the
        PP-parallel DeepGEMM warmup sweeps batch sizes far larger than the decode
        runner's max_bs, and the eager (no graph runner) autotune path has no
        runner buffers -- build one here and hand it to _dummy_run (reused across
        the sweep; _dummy_run slices it per shape). When a decode graph runner
        exists, the flashinfer autotune instead reuses that runner's own buffers.
        """
        mr = self.model_runner
        return _allocate_decode_buffers(
            device=mr.device,
            max_bs=max_bs,
            max_num_token=max_bs * num_tokens_per_bs,
            hidden_size=mr.model_config.hidden_size,
            vocab_size=mr.model_config.vocab_size,
            dtype=mr.model_config.dtype,
            dp_size=mr.server_args.dp_size,
            pp_size=mr.server_args.pp_size,
            is_encoder_decoder=mr.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather(mr.server_args),
            seq_len_fill_value=mr.attn_backend.get_cuda_graph_seq_len_fill_value(),
            encoder_len_fill_value=(
                getattr(mr.model_config.hf_config, "max_source_positions", 0)
                if mr.model_config.is_encoder_decoder
                else 0
            ),
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            hc_hidden_size=getattr(mr.model_config, "hc_hidden_size", None),
        )

    def _dummy_run(
        self,
        batch_size: int,
        run_ctx=None,
        forward_mode_override: Optional[ForwardMode] = None,
        *,
        buffers,
    ):
        """Run a dummy forward pass for warmup/profiling.

        forward_mode_override forces EXTEND/DECODE regardless of
        is_generation (used by the PP-parallel DeepGEMM warmup).

        buffers: a prepared static decode-buffer set, sized >= this dummy shape,
        which _dummy_run slices to (batch_size, num_tokens). The caller owns the
        shape and the allocation -- the flashinfer autotune reuses the decode
        runner's buffers (at its captured max_bs) or, in the eager path, a set
        from _alloc_dummy_decode_buffers; the PP-DeepGEMM warmup builds one via
        _alloc_dummy_decode_buffers. _dummy_run never allocates and never re-pads
        (autotune must run at the captured shape; the PP warmup pre-pads and
        sizes its buffer to match).
        """
        mr = self.model_runner
        if forward_mode_override is not None:
            capture_forward_mode = forward_mode_override
        elif mr.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if mr.spec_algorithm.is_speculative():
            if mr.is_draft_worker:
                if not mr.spec_algorithm.supports_target_verify_for_draft():
                    raise RuntimeError("This should not happen")
            capture_forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens_per_bs = (
                mr.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                    mr.server_args.speculative_num_draft_tokens, mr.is_draft_worker
                )
            )

        if mr.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        # The caller owns the shape: it passes a prepared static buffer sized
        # >= this dummy shape, which we slice below. _dummy_run neither allocates
        # nor re-pads -- autotune must run at the runner's captured shape (re-
        # padding could overflow the reused buffers), and the PP-DeepGEMM warmup
        # applies its own MLP-sync padding before sizing its buffer.
        assert (
            buffers is not None
            and num_tokens <= buffers.input_ids.shape[0]
            and batch_size <= buffers.seq_lens.shape[0]
        ), (
            f"_dummy_run needs a static buffer >= (num_tokens={num_tokens}, "
            f"batch_size={batch_size}); got "
            + (
                "None"
                if buffers is None
                else f"(input_ids={buffers.input_ids.shape[0]}, "
                f"seq_lens={buffers.seq_lens.shape[0]})"
            )
        )

        seq_len_fill_value = mr.attn_backend.get_cuda_graph_seq_len_fill_value()

        if mr.server_args.enable_torch_compile:
            set_torch_compile_config()
            should_disable_torch_compile = not getattr(
                mr.model, "_can_torch_compile", True
            )
            if should_disable_torch_compile:
                log_info_on_rank0(
                    logger,
                    "Transformers backend model reports it is not torch.compile "
                    "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
                )
                mr.server_args.enable_torch_compile = False

        # NOTE: aux hidden state capture (eagle3/dflash) is already
        # configured by init_aux_hidden_state_capture() in initialize().

        require_mlp_tp_gather_ = require_mlp_tp_gather(mr.server_args)
        if require_gathered_buffer(mr.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(mr.server_args)

        # Slice the (possibly larger, reused) static buffers to this dummy shape.
        # The runner repopulates every field at capture/reserve, and the PP
        # warmup discards its buffer, so the writes below (and the metadata init)
        # are transient.
        input_ids = buffers.input_ids[:num_tokens]
        positions = buffers.positions[:num_tokens]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        req_pool_indices = buffers.req_pool_indices[:batch_size]
        seq_lens = buffers.seq_lens[:batch_size]
        seq_lens_cpu = buffers.seq_lens_cpu[:batch_size]
        encoder_lens = (
            buffers.encoder_lens[:batch_size]
            if buffers.encoder_lens is not None
            else None
        )

        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        if capture_forward_mode == ForwardMode.EXTEND:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=mr.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=mr.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=mr.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if mr.server_args.pp_size > 1:
            # PP0 already cp-split hidden_states before send.
            pp_hidden_tokens = num_tokens
            if (
                capture_forward_mode == ForwardMode.EXTEND
                and mr.pp_rank != 0
                and mr.attn_cp_size > 1
            ):
                pp_hidden_tokens = num_tokens // mr.attn_cp_size
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:pp_hidden_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            global_num_tokens_cpu = [num_tokens] * mr.server_args.dp_size
        elif require_attn_tp_gather(mr.server_args):
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=mr.device
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
        else:
            global_dp_buffer_len = None
            global_num_tokens_cpu = None

        spec_info = create_dummy_verify_input(
            mr.spec_algorithm,
            mr.server_args,
            buffers.custom_mask,
            num_tokens_per_bs,
            mr.is_draft_worker,
        )
        if spec_info is not None and (
            mr.spec_algorithm.is_eagle() or mr.spec_algorithm.is_standalone()
        ):
            # MTP models (e.g. deepseek_nextn) read spec_info.hidden_states
            # during forward; provide a dummy so warmup doesn't crash.
            spec_info.hidden_states = torch.zeros(
                (num_tokens, mr.model_config.hidden_size),
                dtype=mr.dtype,
                device=mr.device,
            )
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if mr.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=mr.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            mr.lora_manager.prepare_lora_batch(forward_batch)

        mr.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                mr.server_args.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(mr.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not mr.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = mr.model.forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(mr.device).synchronize()
        mr.tp_group.barrier()
        with forward_context(ForwardContext(attn_backend=mr.attn_backend)):
            with torch.inference_mode(), run_ctx or empty_context():
                run_once()

    def _autotune_buffers(self) -> Tuple[Optional[Any], Optional[int]]:
        """Static decode buffers + max captured bs for warmup() to hand the
        flashinfer-autotune dummy forward, so it reuses this runner's already-
        allocated buffers instead of allocating a throwaway set.

        Returns (None, None) by default. Only the decode runner overrides this:
        its buffers carry every field the dummy decode forward reads. Prefill
        buffers deliberately do not (no seq_lens / req_pool_indices / logits
        buffer), and the eager runner's grow-on-demand buffers are not sized to
        a single captured bs, so both fall back to a dummy decode set allocated
        in warmup().
        """
        return None, None

    @abstractmethod
    def can_run_graph(self, forward_batch: ForwardBatch) -> bool: ...

    @abstractmethod
    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def execute(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...
