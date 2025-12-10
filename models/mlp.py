from torch import nn
from utils.misc import take_annotation_from
from torch import Tensor
from typing import List, Tuple, Union
from math import sqrt


class MLP(nn.Module):
    """
    Multi-layer perceptron with a separated backbone and classification head.

    :param input_dim: Input dimension.
    :param hidden_dims: List of the hidden dimensions.
    :param output_dim: Output dimension (number of classes).
    :param token_dim: Dimension of the embedding token (input to LLM).
    :param norm: Normalization layer.
    :param activation: Activation function.
    :param dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        token_dim: int = 4096,  # Default to common LLM dim (e.g., Llama/Qwen)
        norm: nn.Module = nn.LayerNorm,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # --- Backbone Construction ---
        layers = []
        
        # Initial hidden layers
        curr_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, dim))
            layers.append(norm(dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            curr_dim = dim

        # Final projection to LLM token dimension
        layers.append(nn.Linear(curr_dim, token_dim))
        layers.append(norm(token_dim))
        layers.append(activation())
        
        self.backbone = nn.Sequential(*layers)

        # --- Classification Head ---
        self.head = nn.Linear(token_dim, output_dim)

        self._initialize_weights()

    def forward(self, x: Tensor, return_embedding: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass.

        :param x: Input tensor of shape (batch_size, input_dim).
        :param return_embedding: If True, returns (logits, embedding).
        :return: Logits or tuple of (logits, embedding).
        """
        embedding = self.backbone(x)
        logits = self.head(embedding)

        if return_embedding:
            return logits, embedding
        
        return logits

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, a=sqrt(5))
                nn.init.zeros_(m.bias.data)