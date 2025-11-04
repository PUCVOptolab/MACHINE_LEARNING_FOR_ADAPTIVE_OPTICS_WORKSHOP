import torch
import math


def cart2pol(x, y):
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)
    return r, theta


def CreateTelescopePupil(Npx: int,
                         shapetype: str,
                         device: str = 'cuda',
                         diameter: float = 1.0) -> torch.Tensor:
    """
    Create a binary pupil mask of size (1,1,Npx,Npx).

    Args:
        Npx:      number of pixels per side
        shapetype:'disc', 'square', 'hex', or others (full aperture)
        device:   torch device
        diameter: pupil diameter as a fraction of full aperture (0 < diameter ≤ 1)

    Returns:
        pupil mask tensor, shape (1, 1, Npx, Npx)
    """
    if not (0 < diameter <= 1):
        raise ValueError("`diameter` must be in the range (0, 1].")

    # normalized grid from -1 to 1
    x = torch.arange(-(Npx - 1) / 2,
                      (Npx - 1) / 2 + 1,
                      step=1,
                      device=device)
    x = x / x.abs().max()
    xv, yv = torch.meshgrid(x, x, indexing='xy')

    # scale coordinates so that a unit circle in scaled coords
    # corresponds to the desired diameter in original coords
    ratio = diameter
    xv_s = xv / ratio
    yv_s = yv / ratio
    r_s  = torch.sqrt(xv_s**2 + yv_s**2)

    # build mask
    if shapetype == "disc":
        pupil = (r_s <= 1).float()
    elif shapetype == "square":
        pupil = ((xv_s.abs() <= 1) & (yv_s.abs() <= 1)).float()
    elif shapetype == "hex":
        pupil = ((xv_s.abs() <= math.sqrt(3) / 2) &
                 (yv_s.abs() <= (xv_s.abs() / math.sqrt(3) + 1))).float()
    else:
        pupil = torch.ones_like(r_s)

    return pupil.unsqueeze(0).unsqueeze(0)  # (1, 1, Npx, Npx)

def make_circular_pupil(N, D, device):
    y = torch.arange(N, device=device) - N//2
    x = torch.arange(N, device=device) - N//2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    return (R <= (D/2)).float()