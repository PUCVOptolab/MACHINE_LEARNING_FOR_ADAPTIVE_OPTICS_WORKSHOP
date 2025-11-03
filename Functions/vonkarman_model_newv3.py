import torch
import math
import numpy as np
from scipy import linalg
from scipy.special import kv, gamma
import torch.nn.functional as F

# === Von Kármán Phase Screen Generator ===
class VonKarmanPhaseScreenGenerator:
    def __init__(self, N=128, D_tel=1, r0=0.1, L0=25, l0=0.01,pupil_mask=None, wavelength=None, batch_size=1, device='cuda', seed=None):
        self.device = device
        self.N = N
        self.D_tel = D_tel
        self.r0 = r0
        self.L0 = L0
        self.l0 = l0
        self.wavelength = wavelength
        self.batch_size = batch_size
        self.seed = seed
        self.pupil_mask = pupil_mask.to(device) if pupil_mask is not None else None
        self._build_grids()

    def _build_grids(self):
        N = self.N
        self.delta = self.D_tel / N
        self.D = N * self.delta

        del_f = 1.0 / (N * self.delta)
        fx = torch.linspace(-N / 2, N / 2 - 1, N, device=self.device) * del_f
        self.fx, self.fy = torch.meshgrid(fx, fx, indexing='xy')
        self.f = torch.sqrt(self.fx**2 + self.fy**2)

        coords = torch.linspace(-self.D / 2, self.D / 2 - self.delta, N, device=self.device)
        x, y = torch.meshgrid(coords, coords, indexing='xy')
        self.x = x.unsqueeze(0).expand(self.batch_size, -1, -1)
        self.y = y.unsqueeze(0).expand(self.batch_size, -1, -1)

    def update_parameters(self, r0=None, L0=None, l0=None, D_tel=None, batch_size=None, pupil_mask=None):
        if r0 is not None: self.r0 = r0
        if L0 is not None: self.L0 = L0
        if l0 is not None: self.l0 = l0
        if D_tel is not None: self.D_tel = D_tel
        if batch_size is not None: self.batch_size = batch_size
        if pupil_mask is not None: self.pupil_mask = pupil_mask.to(self.device)
        self._build_grids()

    def _apply_pupil_mask(self, phs_tensor):
        if self.pupil_mask is not None:
            return phs_tensor * self.pupil_mask
        return phs_tensor

    def _ifft2_batch(self, G, delta_f):
        N = G.shape[-1]
        return torch.fft.ifft2(torch.fft.fftshift(G, dim=(-2, -1)), dim=(-2, -1)) * (N * delta_f) ** 2

    def generate_random_phase(self):
        N = self.N
        r0, L0, l0 = self.r0, self.L0, self.l0
        del_f = 1.0 / (N * self.delta)

        fm = 5.92 / l0 / (2 * math.pi)
        f0 = 1.0 / L0

        PSD_phi = (0.023 * r0 ** (-5. / 3.) *
                   torch.exp(-(self.f / fm) ** 2) /
                   ((self.f ** 2 + f0 ** 2) ** (11. / 6.)))
        PSD_phi[N//2, N//2] = 0
        PSD_phi = PSD_phi.expand(self.batch_size, -1, -1)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        cn = (torch.randn((self.batch_size, N, N), device=self.device) +
              1j * torch.randn((self.batch_size, N, N), device=self.device)) * torch.sqrt(PSD_phi) * del_f

        phs = self._ifft2_batch(cn, 1).real
        if self.wavelength is not None:
            phs = phs * (self.wavelength * 1e9) / (2 * math.pi)

        return self._apply_pupil_mask(phs.unsqueeze(1))

    def generate_subharmonic_phase(self):
        N = self.N
        phs_lo = torch.zeros((self.batch_size, N, N), dtype=torch.cfloat, device=self.device)

        for p in range(1, 4):
            del_f = 1.0 / (3 ** p * self.D)
            base = torch.tensor([-1, 0, 1], device=self.device) * del_f
            fx, fy = torch.meshgrid(base, base, indexing='xy')
            f = torch.sqrt(fx ** 2 + fy ** 2)

            fm = 5.92 / self.l0 / (2 * math.pi)
            f0 = 1.0 / self.L0

            PSD_phi = (0.023 * self.r0 ** (-5. / 3.) *
                       torch.exp(-(f / fm) ** 2) /
                       ((f ** 2 + f0 ** 2) ** (11. / 6.)))
            PSD_phi[1, 1] = 0

            cn = (torch.randn((self.batch_size, 3, 3), device=self.device) +
                  1j * torch.randn((self.batch_size, 3, 3), device=self.device)) * torch.sqrt(PSD_phi)[None, :, :] * del_f

            for i in range(3):
                for j in range(3):
                    phase = 2 * math.pi * (fx[i, j] * self.x + fy[i, j] * self.y)
                    phs_lo += cn[:, i, j].unsqueeze(-1).unsqueeze(-1) * torch.exp(1j * phase)

        phs_lo = phs_lo.real - phs_lo.real.mean(dim=(1, 2), keepdim=True)

        if self.wavelength is not None:
            phs_lo = phs_lo * (self.wavelength * 1e9) / (2 * math.pi)

        return self._apply_pupil_mask(phs_lo.unsqueeze(1))

    def generate_total_phase(self,alpha=0.6,beta=0.4):
        phs_hi = self.generate_random_phase()
        phs_lo = self.generate_subharmonic_phase()
        return (phs_hi*alpha + phs_lo*beta)



# Exact von Kármán covariance (as in AOTools)
def turb_phase_covariance(r, r0, L0):
    r = r + 1e-40
    A  = (L0 / r0)**(5.0/3.0)
    B1 = (2**(-5.0/6.0)) * gamma(11.0/6.0) / (np.pi**(8.0/3.0))
    B2 = ((24.0/5.0) * gamma(6.0/5.0))**(5.0/6.0)
    x = (2.0 * np.pi * r) / L0
    C = x**(5.0/6.0) * kv(5.0/6.0, x)
    C[np.isnan(C)] = ((2.0*np.pi*1e-40)/L0)**(5.0/6.0) * kv(5.0/6.0, (2.0*np.pi*1e-40)/L0)
    return A * B1 * B2 * C

class InfiniteVonKarmanPhaseScreenGenerator:
    def __init__(
        self,
        N=128,
        D_tel=8.0,
        r0=0.15,
        L0=25.0,
        l0=0.01,
        n_columns=2,
        init_phase=None,
        device='cpu',
        seed=None,
        pupil_mask=None,
        wavelength=None,
        wind_dir_deg=0.0,            # ← direction in degress x layer
        wind_speed=20,        # wind speed in m/seg
        fps=1000
    ):
        self.N          = N
        self.D_tel      = D_tel
        self.r0         = r0
        self.L0         = L0
        self.n_columns  = n_columns
        self.device     = device
        self.seed       = seed
        self.wavelength = wavelength
        self.pupil_mask = pupil_mask.to(device) if pupil_mask is not None else None
        self.wind_dir_deg = wind_dir_deg  # store wind direction
        self.wind_speed = wind_speed
        self.fps        = fps

        self.pxm = self.D_tel/self.N
        self.wind_speed_px_it = self.wind_speed/self.pxm/self.fps


        # Persistent RNG on correct device
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)

        # Resolution fit to crop area
        self.final_N = N
        self.scaling_factor = 2/self.wind_speed_px_it
        self.N = int(np.round(self.N*np.sqrt(2)*self.scaling_factor))
        square_pad = torch.ones((1,1,self.N,self.N),device=self.device)
        square_pad = self._rotate_phase(square_pad)
        # Build A/B matrices (NumPy)
        self._build_ab_matrices()
        # Convert A, B to torch tensors on self.device (ensure float32)
        self.A_mat = torch.from_numpy(self.A_mat).float().to(self.device)
        self.B_mat = torch.from_numpy(self.B_mat).float().to(self.device)
        # self.A_mat = torch.from_numpy(self.A_mat).to(self.device)
        # self.B_mat = torch.from_numpy(self.B_mat).to(self.device)

        # Generate initial screen via original PSF-based generator (CPU)
        if init_phase is None:
            init_gen = VonKarmanPhaseScreenGenerator(
                N=self.N, D_tel=D_tel, r0=r0, L0=L0, l0=l0,
                pupil_mask=None, wavelength=None,
                batch_size=1, device='cpu', seed=seed
            )
            init = init_gen.generate_total_phase(alpha=0.5, beta=0.5)
        self._scrn = init.squeeze().to(self.device)  # shape (N, N)

        # Precompute stencil indices
        mask = np.zeros((self.N, self.N)); mask[:n_columns, :] = 1
        coords = np.column_stack(np.where(mask == 1))
        self._rows = torch.from_numpy(coords[:,0]).long().to(self.device)
        self._cols = torch.from_numpy(coords[:,1]).long().to(self.device)


    # add this helper to rotate the 4D tensor [B,1,H,W]
    def _rotate_phase(self, phs: torch.Tensor) -> torch.Tensor:
        angle = self.wind_dir_deg * math.pi / 180.0
        if angle == 0.0:
            return phs
        B, C, H, W = phs.shape
        # build a (B×2×3) rotation matrix
        theta = torch.tensor([
            [ math.cos(angle), -math.sin(angle), 0.0],
            [ math.sin(angle),  math.cos(angle), 0.0]
        ], device=self.device, dtype=phs.dtype)\
        .unsqueeze(0).repeat(B,1,1)
        # create a sampling grid
        grid = F.affine_grid(theta, phs.size(), align_corners=True)
        # rotate (we use border padding so edges are replicated)
        return F.grid_sample(phs, grid,
                             mode='bilinear',
                             padding_mode='zeros',
                             align_corners=True)

    def crop_to_final(self, phs: torch.Tensor) -> torch.Tensor:
        """
        Center-crop a (B, C, M, M) tensor to (B, C, N, N),
        where M = self.M and N = self.final_N.
        """
        M = int(self.N/self.scaling_factor)
        N = self.final_N

        # how many pixels to remove total, and offset on each side
        pad = M - N
        off = pad // 2

        # slice out the central N×N patch
        return phs[:, :, off:off+N, off:off+N]

    def downsample_tensor(self,x: torch.Tensor,
                scale_factor: float = None,
                out_size: tuple[int,int] = None,
                mode: str = 'bilinear') -> torch.Tensor:
        """
        Down-sample a 4D tensor (B, C, H, W) by interpolation.
        Either `scale_factor` (float < 1) or `out_size` (H_out, W_out) must be provided.
        mode: 'nearest' | 'bilinear' | 'bicubic' | etc.
        """
        if (scale_factor is None) == (out_size is None):
            raise ValueError("Specify exactly one of scale_factor or out_size")
        return F.interpolate(
            x,
            size=out_size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=(mode in ('linear', 'bilinear', 'bicubic'))  # align only for real‐valued modes
        )


    def _build_ab_matrices(self):
        pix = self.D_tel / self.N
        mask = np.zeros((self.N, self.N)); mask[:self.n_columns, :] = 1
        coords = np.column_stack(np.where(mask == 1))
        pos_z = coords * pix
        X = np.zeros((self.N, 2)); X[:,0] = -1; X[:,1] = np.arange(self.N)
        pos_x = X * pix
        allpos = np.vstack((pos_z, pos_x))
        d = allpos[:, None, :] - allpos[None, :, :]
        r = np.linalg.norm(d, axis=2)

        cov = turb_phase_covariance(r, self.r0, self.L0)
        n = coords.shape[0]
        Czz = cov[:n, :n]; Czx = cov[:n, n:]
        Cxz = cov[n:, :n]; Cxx = cov[n:, n:]

        try:
            cf = linalg.cho_factor(Czz)
            invCzz = linalg.cho_solve(cf, np.eye(n))
        except linalg.LinAlgError:
            invCzz = np.linalg.pinv(Czz)
        self.A_mat = Cxz.dot(invCzz)

        BBt = Cxx - self.A_mat.dot(Czx)
        U, W, _ = np.linalg.svd(BBt)
        self.B_mat = U.dot(np.diag(np.sqrt(W)))

    def _apply_pupil_mask(self, phs_tensor):
        if self.pupil_mask is not None:
            return phs_tensor * self.pupil_mask
        return phs_tensor        

    def evolve(self):
        # Draw a new Gaussian vector on the correct device
        b = torch.randn(self.N, generator=self._rng, device=self.device)

        # Gather stencil values
        zs = self._scrn[self._rows, self._cols]

        # Compute the new row
        X = self.A_mat.matmul(zs) + self.B_mat.matmul(b)
        new_row = X.unsqueeze(0)

        # Prepend and truncate
        self._scrn = torch.cat([new_row, self._scrn], dim=0)[:self.N, :]

        # Expand to output shape
        phs = self._scrn.unsqueeze(0).unsqueeze(0)
        if self.wavelength is not None:
            phs = phs * (self.wavelength * 1e9) / (2 * torch.pi)

        phs = self.downsample_tensor(phs,scale_factor=1/self.scaling_factor)
        phs = self._rotate_phase(phs)
        phs = self.crop_to_final(phs)    
        return self._apply_pupil_mask(phs)





class MultiLayerPhaseScreen:
    def __init__(
        self,
        layers: list,
        N: int = 128,
        D_tel: float = 8.0,
        device: str = 'cuda',
        pupil_mask=None,
        wavelength=None,
        seed=None,
    ):
        """
        layers: list of dicts, each with:
          - r0 (float): Fried parameter
          - L0 (float): outer scale
          - l0 (optional float): inner scale, default 0.01
          - wind_speed (float): m/s
          - wind_dir_deg (float)
          - dt (float): s per step
          - Cn2 (optional float): weight, default 1.0
          - altitude (float): layer height in m
        """
        self.device     = device
        self.pupil_mask = pupil_mask.to(device) if pupil_mask is not None else None
        self.wavelength = wavelength

        # pixel scale at pupil
        self.delta = D_tel / N

        # build per-layer generators
        self.layers = []
        for idx, info in enumerate(layers):
            layer_seed = None if seed is None else (seed + idx)
            gen = InfiniteVonKarmanPhaseScreenGenerator(
                N=N,
                D_tel=D_tel,
                r0=info['r0'],
                L0=info['L0'],
                l0=info.get('l0', 0.01),
                n_columns=info.get('n_columns', 2),
                device=device,
                seed=layer_seed,
                pupil_mask=self.pupil_mask,
                wavelength=None,
                wind_dir_deg=info['wind_dir_deg'],
                wind_speed=info.get('wind_speed',20),
                fps=info.get('fps',500)
            )
            gen.Cn2      = info.get('Cn2', 1.0)
            self.layers.append(gen)

    def evolve(self):
        """
        Step each layer, apply wind drift, then sum phases.
        If angles_rad is provided (list of float radians), returns a tensor of shape
        (batch_size, len(angles_rad), N, N) with anisoplanatic shifts per direction.
        Otherwise returns (batch_size, 1, N, N) for on-axis.
        """
        # accumulate per-angle total
        phs_out = []
        total = None
        for gen in self.layers:
            phs_layer = gen.evolve()  # (batch_size, N, N)
            # weight by Cn2
            phs_layer = phs_layer * gen.Cn2
            total = phs_layer if total is None else total + phs_layer
        # expand channel dimension
        phs_out.append(total)  # (batch_size,1,N,N)

        # stack along angle dimension
        result = torch.cat(phs_out, dim=1)  # (batch_size, M, N, N)

        # convert to nm if requested
        if self.wavelength is not None:
            result = result * (self.wavelength * 1e9) / (2 * torch.pi)
        # apply pupil mask
        if self.pupil_mask is not None:
            result = result * self.pupil_mask

        return result
