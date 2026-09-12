from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin


@ModelBase.register("AgnesForConditionalGeneration")
class AgnesTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Agnes 3.0 text model with hybrid attention and a second FFN branch."""

    model_arch = gguf.MODEL_ARCH.QWEN35

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        name = name.replace(".delta_attn.", ".linear_attn.").replace(".global_attn.", ".self_attn.")
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        # Normalize Agnes names for the Qwen3.5 parameter writer.
        layer_types = self.hparams.get("layer_types")
        if layer_types is not None:
            self.hparams["layer_types"] = [
                "linear_attention" if t == "agnes_delta_attention" else "full_attention"
                for t in layer_types
            ]
        if "full_attention_interval" not in self.hparams and "global_attention_interval" in self.hparams:
            self.hparams["full_attention_interval"] = self.hparams["global_attention_interval"]

        super().set_gguf_parameters()

        n_ff_par = self.hparams.get("parallel_ffn_intermediate_size")
        if n_ff_par:
            logger.info("parallel FFN length: %d", n_ff_par)
            self.gguf_writer.add_feed_forward_parallel_length(int(n_ff_par))
