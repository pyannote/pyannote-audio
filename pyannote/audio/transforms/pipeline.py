from __future__ import annotations

from torch.nn import Sequential


class TransformsPipe(Sequential):
    """
    Transforms that are probabalistic, can be returned for validation
    and testing with the `validate_pipe` command.
    """

    def validate_pipe(self) -> TransformsPipe:
        return TransformsPipe(*[a for a in self if not hasattr(a, "p") or a.p < 1])
