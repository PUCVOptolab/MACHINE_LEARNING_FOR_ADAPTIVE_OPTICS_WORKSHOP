import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from Functions.fourier_masks import *
from Functions.pupils import *


def update_namespace_defaults(par):
    """
    Update an argparse.Namespace with default values for missing parameters.
    
    This function assumes par is an argparse.Namespace (or None).
    If par is None, a new empty namespace is created.
    
    The following keys are ensured to exist:
       - visLevel (default: 1)
       - binning_factor (default: 1)
       - N_subapertures (default: 128)
       - N_points_aperture (default: N_subapertures * binning_factor)
       - n_pix_subap (default: N_points_aperture / N_subapertures)
       - photon_flux (default: 10000)
       - psf_sampling (default: 4)
       - N_fourier_points (default: int(psf_sampling * N_points_aperture))
       - wavelength (default: 0.635e-6)
    
    Parameters:
        par: an argparse.Namespace or None.
    
    Returns:
        Updated namespace with missing parameters added.
    """
    from argparse import Namespace
    if par is None:
        par = Namespace()
    
    if not hasattr(par, 'visLevel'):
        par.visLevel = 1
    if not hasattr(par, 'wfs_type'):
        par.wfs_type = 'PWFS'
    if not hasattr(par, 'binning_factor'):
        par.binning_factor = 1
    if not hasattr(par, 'Npix'):
        par.Npix = 128
    # Compute dependent parameters—make sure the required parameters exist first.
    if not hasattr(par, 'photon_flux'):
        par.photon_flux = 1
    if not hasattr(par, 'psf_sampling'):
        par.psf_sampling = 4
    if not hasattr(par, 'wavelength'):
        par.wavelength = None
    if not hasattr(par, 'calibration_gain'):
        par.calibration_gain = 0.7

    return par


class MVM:
    def __init__(self, aperture,FourierFilter,jModes=[2,3,4], par=None, device='cpu'):
        """
        Constructor for the wavefront reconstructor.
        
        Parameters:
          - wavefront_rad: input wavefront tensor
          - par: dictionary with system parameters (if None, default values are used)
          - parRec: dictionary with reconstruction parameters (if None, default values are used)
          - FourierFilter: custom Fourier mask; if None, genOTF_oomao() is used to create one
          - verbose: Boolean; if True, RMS values are stored during iteration.
          - device: torch.device to use (if None, automatically set to CUDA if available)
        """
        super().__init__()


        # Create pupil mask from the input wavefront (nonzero values define the pupil)
        self.Aperture = aperture        
        self.FourierFilter = FourierFilter
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Np = self.Aperture.shape[-1]
        self.Nf = self.FourierFilter.shape[-1]
        self.Fsamp = self.Nf//self.Np   #Fourier sampling ratio
        # Update namespace defaults (if par is None or missing fields, they're added).
        par = update_namespace_defaults(par)
        self.par = par
        
        self.jModes = jModes
        self.calibration_gain = par.calibration_gain
        self.photon_flux = par.photon_flux
        self.N_fourier_points = FourierFilter.shape[-1]
        self.N_points_aperture = aperture.shape[-1]

        # create modes
        self.z_stack,self.z_tensor = self.CreateZernikePolynomials(Npix=self.Np,
                                                                   jModes=self.jModes,
                                                                   device=self.device)

        # Flat
        self.flat_stack,self.flat_tensor = self.CreateZernikePolynomials(Npix=self.Np,
                                                                   jModes=[0],
                                                                   device=self.device)

        
        # Create coordinate grids and related attributes
        Npa = self.N_points_aperture
        self.x = torch.linspace(-1 + 1/Npa, 1 - 1/Npa, Npa, device=self.device)
        self.y = torch.linspace(-1 + 1/Npa, 1 - 1/Npa, Npa, device=self.device)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')
        self.grid = (self.X + self.Y) / 2
        self.tilt = -(math.pi / self.Fsamp) * self.grid
        self.tilt = self.tilt.unsqueeze(0).unsqueeze(0).to(self.device)  # shape (1,1,Npa,Npa)

        # Define zero-padding indices (using 0-indexing)
        ind_start = int(self.N_fourier_points / 2 - Npa / 2)
        ind_end = ind_start + Npa
        self.ind_zp = slice(ind_start, ind_end)

        self.sqrt_photon_flux = math.sqrt(self.photon_flux)
        self.photon_factor = (self.sqrt_photon_flux * self.Np) / self.N_points_aperture



        # Define a starting guess (e.g., flat wavefront, zeros)
        self.starting_value = torch.zeros((1, 1, Npa, Npa), device=self.device)

        # Calibrate
        self.I_0 = self.linear_forward_model(self.flat_tensor)
        self.calibrate_wfs(self.z_tensor)
        


    # ---------------------
    # Helper Methods
    # ---------------------
    def cart2pol_torch(self,x, y):
        rho = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        return rho, phi


    def CreateZernikePolynomials(self,
        Npix: int = 128,
        jModes=None,
        pupil=None,
        device: str = "cpu"
    ):
        """
        Generate Zernike modes over a full N×N grid.

        Returns
        -------
        zmatrix : torch.Tensor, shape (N*N, len(jModes))
            Each column is one flattened Zernike mode.
        ztensor : torch.Tensor, shape (len(jModes), 1, N, N)
            Each mode reshaped into a 1×N×N image.
        """
        device = torch.device(device)
        if jModes is None:
            raise ValueError("Provide jModes (list of Noll indices).")
        jModes = list(jModes)
        nmodes = len(jModes)

        # pupil mask
        if pupil is None:
            pupil_t = CreateTelescopePupil(Npix,shapetype="disc", device=device)
        else:
            pupil_t = torch.as_tensor(pupil, device=device).float()
        pupil_flat = pupil_t.flatten()  # shape (N*N,)

        # normalized coordinates
        u = torch.linspace(-1, 1, Npix, device=device)
        xv, yv = torch.meshgrid(u, u, indexing='xy')
        r_full, o_full = self.cart2pol_torch(xv, yv)
        r_flat = r_full.flatten()
        o_flat = o_full.flatten()

        # prepare output
        zmatrix = torch.zeros((Npix**2, nmodes), device=device, dtype=torch.float32)

        # radial polynomial builder
        def R_fun(r, n, m):
            R = torch.zeros_like(r)
            # number of terms
            max_s = (n - m) // 2
            for s in range(max_s + 1):
                # compute coefficient in Python
                num = (-1)**s * math.prod(range(1, n - s + 1))
                den = (
                    math.prod(range(1, s + 1)) *
                    math.prod(range(1, (n + m)//2 - s + 1)) *
                    math.prod(range(1, (n - m)//2 - s + 1))
                )
                R = R + (num / den) * r.pow(n - 2*s)
            return R

        # build each mode
        for i, j in enumerate(jModes):
            # get (n,m) from Noll index
            n = 0
            m = 0
            j1 = j - 1
            while j1 > n:
                n += 1
                j1 -= n
                m = (-1)**j * (n % 2 + 2 * math.floor((j1 + (n+1)%2)/2))
            m = abs(m)

            # radial part
            R = R_fun(r_flat, n, m)
            scale = math.sqrt(n + 1)

            if m == 0:
                mode_flat = scale * R
            else:
                factor = scale * math.sqrt(2)
                if (j % 2) == 0:
                    mode_flat = factor * R * torch.cos(m * o_flat)
                else:
                    mode_flat = factor * R * torch.sin(m * o_flat)

            # zero outside pupil
            mode_flat = mode_flat * pupil_flat
            zmatrix[:, i] = mode_flat

        # also return as a 4D tensor: (nmodes, 1, N, N)
        ztensor = zmatrix.t().view(nmodes, 1, Npix, Npix)

        return zmatrix, ztensor



    def _matlab_flip(self, sensor_data):
        temp = sensor_data.transpose(-1, -2)
        temp = torch.flip(temp, dims=[-1])
        temp = temp.transpose(-1, -2)
        temp = torch.flip(temp, dims=[-1])
        return temp.transpose(-1, -2)

    def sum_normalization(self, sensor_data):
        normalized_data = sensor_data/torch.sum(sensor_data,dim=(2,3),keepdim=True)
        return normalized_data

    def _remove_piston(self, phi):
        mask = self.Aperture
        sum_val = (phi * mask).sum(dim=(-2, -1), keepdim=True)
        count = mask.sum(dim=(-2, -1), keepdim=True)
        phi_piston = sum_val / (count + (count == 0).float())
        return mask * (phi - phi_piston)

    def _FFT2(self, x, n, norm_mode='ortho'):
        if norm_mode == 'backward':
            return torch.fft.fft2(x, s=(n, n), dim=(-2, -1))
        elif norm_mode == 'forward':
            fft_forward = torch.fft.fft2(x, s=(n, n), dim=(-2, -1), norm='forward')
            return fft_forward * (n * n)
        elif norm_mode == 'ortho':
            fft_ortho = torch.fft.fft2(x, s=(n, n), dim=(-2, -1), norm='ortho')
            return fft_ortho * n
        else:
            raise ValueError("Invalid norm_mode. Choose 'backward', 'forward', or 'ortho'.")

    def _IFFT2(self, x, n):
        return torch.fft.ifft2(x, s=(n, n), dim=(-2, -1))

    def _f_zeropad(self, wavefront):
        pad_total = self.N_fourier_points - wavefront.shape[-1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # Pad last two dimensions; F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        return F.pad(wavefront, (pad_left, pad_right, pad_left, pad_right))    

    def _f_getRMS(self, f):
        B = f.shape[0]
        f_flat = f.contiguous().view(B, -1)
        mask = f_flat != 0
        count = mask.sum(dim=1)
        sum_sq = (torch.abs(f_flat)**2 * mask.float()).sum(dim=1)
        rms = torch.sqrt(sum_sq / count.float().clamp_min(1))
        return torch.where(count > 0, rms, torch.zeros_like(rms))

    def _f_getl2_norm(self, f):
        return torch.sqrt(torch.sum(torch.abs(f)**2, dim=tuple(range(1, f.ndim))))

    def _gaussian_kernel_2d(self, sigma, kernel_size):
        ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=self.device, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        return kernel / kernel.sum()


    def generate_full_frame(self,wavefront_rad):
        """
        Compute the full-frame intensity from the current wavefront.
        """
        wavefront_tilted = wavefront_rad + self.tilt
        field = self.Aperture * (1.0 / (self.N_fourier_points**2)) * self.photon_factor * torch.exp(1j * wavefront_tilted)
        field_zp = self._f_zeropad(field)
        FT2_field = self._FFT2(field_zp, self.N_fourier_points)
        filtered_field = FT2_field * self.FourierFilter
        field2 = self._FFT2(filtered_field, self.N_fourier_points)
        intensity_octopus = field2 * torch.conj(field2)
        return torch.real(intensity_octopus).transpose(-2, -1)


    def update_fouriermask(self, new_mask):
        """
        Update the Fourier mask used in the reconstruction.
        
        Parameters:
            new_mask (torch.Tensor): A tensor containing the updated Fourier mask.
            
        Raises:
            ValueError: If the shape of the new mask does not match the current mask.
        """
        if new_mask.shape != self.FourierFilter.shape:
            raise ValueError("New Fourier mask shape does not match the current Fourier mask shape.")
        self.FourierFilter = new_mask.to(self.device)
        self.calibrate_wfs(self.z_tensor)


    def linear_forward_model(self,wavefront_rad):
        I_tensor = self.generate_full_frame(wavefront_rad)
        I_tensor = self._matlab_flip(I_tensor)  
        return I_tensor    

    def calibrate_wfs(self,z_tensor):
        sp = self.linear_forward_model(z_tensor*self.calibration_gain)
        sp = self.sum_normalization(sp)   
        sm = self.linear_forward_model(-z_tensor*self.calibration_gain)
        sm = self.sum_normalization(sm) 

        IM_tensor = 0.5*(sp-sm)/self.calibration_gain
        nmodes,_,Npix,Npix = IM_tensor.shape
        interaction_matrix = IM_tensor.view(nmodes, Npix**2)
        self.interaction_matrix = interaction_matrix.permute(1,0)
        self.control_matrix = torch.linalg.pinv(self.interaction_matrix) 
        self.conditional_matrix = self.interaction_matrix.T @ self.interaction_matrix
        #return self.interaction_matrix, self.control_matrix, self.conditional_matrix   

    def zern2phase(self, z):
        """
        Reconstruct the phase map and Zernike coefficients from sensor image batch.
        """
        B, _ = z.shape
        # phi_vect: (B, N^2)
        phi_vect = torch.einsum('ij,bj->bi', self.z_stack, z)

        phi_est = phi_vect.view(B, 1, self.Np, self.Np)
        return phi_est   


    def phase2zern(self, wavefront):
        """
        Reconstruct the phase map and Zernike coefficients from sensor image batch.
        """
        B, _,Np,Np = wavefront.shape
        wavefront_flat = wavefront.view(B, Np * Np)
        # phi_vect: (B, N^2)
        phase_decon_matc = torch.linalg.pinv(self.z_stack)
        z_est = torch.einsum('ij,bj->bi', phase_decon_matc, wavefront_flat)

        # phi_vect: (B, N^2)
        phi_vect = torch.einsum('ij,bj->bi', self.z_stack, z_est)

        phi_est = phi_vect.view(B, 1, self.Np, self.Np)
        return z_est,phi_est  


    def run_method(self, I, true_phi=None,initphase=None):
        """
        Reconstruct the phase map and Zernike coefficients from sensor image batch.
        
        Parameters:
            I (Tensor): shape (B, 1, N, N) — normalized sensor images.
            
        Returns:
            phi_est (Tensor): shape (B, 1, Np, Np) — reconstructed phase maps.
            z_est  (Tensor): shape (B, Nmodes) — Zernike coefficients.
        """
        I = self.sum_normalization(I)
        I = I-self.sum_normalization(self.I_0)
        B, _, N, N = I.shape
        I_flat = I.view(B, N * N)
        metrics= {}

        # Expand control matrix to (B, Nmodes, N^2) to apply via bmm
        # Alternatively use einsum for clarity
        # z_est: (B, Nmodes)
        z_est = torch.einsum('ij,bj->bi', self.control_matrix, I_flat)

        # phi_vect: (B, N^2)
        phi_vect = torch.einsum('ij,bj->bi', self.z_stack, z_est)

        phi_est = phi_vect.view(B, 1, self.Np, self.Np)
        metrics['Z_coefs'] = z_est
        return phi_est, metrics         







# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     Npix = 128
#     Zj = [2,200]
#     jmodes = np.arange(Zj[0], Zj[1]-1) 

#     pwfs = genOTF_PWFS4(
#         N_fourier_points=4*Npix,
#         N_points_aperture=Npix,
#         device=device)

#     pupil = CreateTelescopePupil(Npix,shapetype="disc", device=device)
 
#     mvm = MVM(aperture=pupil,FourierFilter=pwfs,jModes=jmodes,device=device)

#     ## Test model
#     from OOMAO_build.vonkarman_model_new  import VonKarmanPhaseScreenGenerator

#     atmSim = VonKarmanPhaseScreenGenerator(N=Npix, D_tel=1, r0=0.90, L0=25, l0=0.01,
#             wavelength=None, batch_size=3, device='cuda',
#             pupil_mask=pupil)

#     phaseMap = atmSim.generate_total_phase()
#     I = mvm.linear_forward_model(phaseMap)

#     phi_est,z_est = mvm.run_method(I)

#     print('done')