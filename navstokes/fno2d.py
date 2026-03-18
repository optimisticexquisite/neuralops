import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D Fourier layer with truncated complex modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, size_x, size_y = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            size_x,
            size_y // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights
        )

        x = torch.fft.irfft2(out_ft, s=(size_x, size_y))
        return x


class FNO2d(nn.Module):
    """FNO-2D block for one-step vorticity prediction."""

    def __init__(
        self,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 48,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
    ):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels + 2, width, kernel_size=1)

        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])

        self.fc1 = nn.Conv2d(width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.activation = nn.GELU()

    @staticmethod
    def _grid(batch: int, size_x: int, size_y: int, device: torch.device) -> torch.Tensor:
        x = torch.linspace(0, 1, size_x, device=device)
        y = torch.linspace(0, 1, size_y, device=device)
        gridx = x.view(1, 1, size_x, 1).repeat(batch, 1, 1, size_y)
        gridy = y.view(1, 1, 1, size_y).repeat(batch, 1, size_x, 1)
        return torch.cat([gridx, gridy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, sx, sy = x.shape
        grid = self._grid(batch, sx, sy, x.device)
        x = torch.cat([x, grid], dim=1)

        x = self.fc0(x)
        for spectral, w in zip(self.spectral_layers, self.w_layers):
            x = self.activation(spectral(x) + w(x))

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
