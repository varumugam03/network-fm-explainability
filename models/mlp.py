from torch import nn
from utils.misc import take_annotation_from
from torch import Tensor
from typing import List
from math import sqrt


class MLP(nn.Module):
    """
    Multi-layer perceptron.

    :param input_dim: Input dimension.
    :param hidden_dims: List of the hidden dimensions.
    :param output_dim: Output dimension.
    :param norm: Normalization layer.
    :param activation: Activation function.
    :param dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        norm: nn.Module = nn.LayerNorm,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = []

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(norm(dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            input_dim = dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the multi-layer perceptron.

        :param x: Input tensor of shape (batch_size, input_dim).
        :return: Output tensor of shape (batch_size, output_dim).
        """

        return self.mlp(x)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self):
        """
        Initialize the weights of the multi-layer perceptron.
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, a=sqrt(5))
                nn.init.zeros_(m.bias.data)
