from typing import Any, Optional

from torch import nn


class LayerFactory:
    """Factory class for non linearity layers.

    Args:
        cls: non linearity class.
        init_args: positional args for cls.
        init_kwargs: keyword args for cls.
    """

    def __init__(
        self,
        cls: type[nn.Module],
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.cls = cls
        self.args = args or []
        self.kwargs = kwargs or {}

    def __call__(self, *args, **kwargs: Any) -> Any:
        """Gets the instantiated object."""
        return self.layer(self, *args, **kwargs)

    def layer(self, **kwargs) -> nn.Module:
        """Instantiated object."""
        kwargs = {**self.kwargs, **kwargs}
        return self.cls(*self.args, **kwargs)
