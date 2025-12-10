from typing import Union

from .mlp import MLP
from .explainer import NetworkExplainer

Model = Union[MLP, NetworkExplainer]

__all__ = [MLP, NetworkExplainer, Model]
