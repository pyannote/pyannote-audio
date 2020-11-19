from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import Sequential

if TYPE_CHECKING:
    from torch import Tensor


class TransformsPipe(Sequential):
    """
    A Sequential with transforms which are applied to data such as a Spectrogram
    Transforms that are probabalistic, can be returned for validation
    and testing with the `validate_pipe` command.
    """

    def validate_pipe(self) -> TransformsPipe:
        return TransformsPipe(*[a for a in self if not hasattr(a, "p") or a.p < 1])

    def spectrogram_pipe(self) -> TransformsPipe:
        return TransformsPipe(
            *[a for a in self if hasattr(a, "_spectrogram") if a._spectrogram]
        )

    def waveform_pipe(self) -> TransformsPipe:
        return TransformsPipe(
            *[a for a in self if not hasattr(a, "_spectrogram") or not a._spectrogram]
        )

    def forward(self, x: Tensor) -> Tensor:
        for module in self:
            x = module(x)
        return x
