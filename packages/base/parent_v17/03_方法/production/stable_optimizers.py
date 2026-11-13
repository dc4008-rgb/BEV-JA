"""PyTorch Adam with float64 working copies and persistent moment state."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


class Float64Adam(torch.optim.Adam):
    def __init__(
        self, parameters: Iterable[torch.nn.Parameter], **kwargs: Any
    ) -> None:
        self.model_parameters = list(parameters)
        self.working_parameters = [
            torch.nn.Parameter(parameter.detach().double().clone())
            for parameter in self.model_parameters
        ]
        super().__init__(self.working_parameters, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for parameter in self.model_parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.detach_()
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for parameter, working in zip(self.model_parameters, self.working_parameters):
            working.copy_(parameter)
            working.grad = (
                None if parameter.grad is None else parameter.grad.detach().double().clone()
            )
            if not torch.isfinite(working).all():
                raise RuntimeError("Adam received non-finite parameters")
            if working.grad is not None and not torch.isfinite(working.grad).all():
                raise RuntimeError("Adam received non-finite gradients")

        super().step()

        for parameter, working in zip(self.model_parameters, self.working_parameters):
            if not torch.isfinite(working).all():
                raise RuntimeError("Adam produced non-finite parameters")
            for name, value in self.state[working].items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise RuntimeError(f"Adam produced non-finite state: {name}")
            if not torch.isfinite(working.to(dtype=parameter.dtype)).all():
                raise RuntimeError("Adam update is not representable in the carrier dtype")
        for parameter, working in zip(self.model_parameters, self.working_parameters):
            parameter.copy_(working)
        return loss