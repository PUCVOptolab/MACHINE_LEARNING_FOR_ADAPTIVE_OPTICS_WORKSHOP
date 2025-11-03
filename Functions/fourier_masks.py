import math
import torch




def genOTF_PWFS4(
    N_fourier_points: int = 512,
    N_points_aperture: int = 40,
    separation: float = 2.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Build the 4-facet pyramid WFS Fourier filter.

    Parameters:
      N_fourier_points: FFT grid size per axis.
      N_points_aperture: Pupil diameter in grid samples.
      separation: Tilt strength for the four quadrants.
      device: “cpu” or “cuda:…”.

    Returns:
      Complex64 tensor of shape (1,1,N_fourier_points,N_fourier_points).
    """
    beta = separation * math.pi / (N_fourier_points / N_points_aperture)
    x = torch.arange(N_fourier_points, device=device, dtype=torch.float32)
    center = N_fourier_points // 2
    fx = (x - center) * (2.0 * center / N_fourier_points)
    FX, FY = torch.meshgrid(fx, fx, indexing='ij')

    H = lambda t: torch.heaviside(t, torch.tensor(0.5, device=device))
    pym = 0j * FX  # complex accumulator

    for sx, sy in ((1,1),(1,-1),(-1,-1),(-1,1)):
        mask = H(sx*FX) * H(sy*FY)
        phase = -beta * (sx*FX + sy*FY)
        pym = pym + mask * torch.exp(1j * phase)

    otf = torch.fft.fftshift(pym).unsqueeze(0).unsqueeze(0)
    return otf.to(device)