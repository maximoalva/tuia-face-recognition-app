import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    """Dos convoluciones 3x3 con BatchNorm y ReLU, y un max pooling que reduce el mapa a la mitad."""
    return nn.Sequential(
        # Sin bias: el BatchNorm que sigue ya suma su propio desplazamiento
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """CNN que recibe un rostro alineado de 112x112 y devuelve un embedding de norma 1."""

    def __init__(self, channels=(32, 64, 128, 256), embedding_dim=128, dropout=0.2):
        super().__init__()
        in_channels = [3, *channels[:-1]]
        self.blocks = nn.Sequential(*[conv_block(i, o) for i, o in zip(in_channels, channels)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.embedding(self.dropout(x))
        return F.normalize(x, dim=1)
