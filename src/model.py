"""
src/model.py
------------
Classe StellarClassifier extraída do notebook principal para facilitar
importação e reutilização em outros scripts ou experimentos.
"""

import torch
import torch.nn as nn


class StellarClassifier(nn.Module):
    """
    Rede neural modular para classificação de corpos celestes.

    Parameters
    ----------
    input_size : int
        Número de features de entrada.
    hidden_sizes : list[int]
        Lista com o número de neurônios em cada camada oculta.
        Ex.: [128, 64] cria duas camadas ocultas.
    num_classes : int
        Número de classes de saída (3 para GALAXY, QSO, STAR).
    dropout_rate : float, optional
        Taxa de Dropout aplicada após cada camada oculta. 0.0 desativa.

    Examples
    --------
    >>> model = StellarClassifier(input_size=10, hidden_sizes=[128, 64], num_classes=3, dropout_rate=0.25)
    >>> x = torch.randn(32, 10)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([32, 3])
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        num_classes: int,
        dropout_rate: float = 0.0,
    ):
        super(StellarClassifier, self).__init__()

        layers = []
        in_features = input_size

        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_features, h_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = h_size

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
