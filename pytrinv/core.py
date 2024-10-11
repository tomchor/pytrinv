import numpy as np
import xarray as xr
from numpy.linalg import pinv
from dask.diagnostics import ProgressBar
from colorama import Fore, Back, Style

def invert_gradient_tensor(ds, test_propagation=True):
    """ Invert gradient matrix using a Pseudo-Inverse algorithm """
    ds = ds.chunk(α=-1, i=-1, j=-1)
    ds["∇ⱼτ̄ᵅ|ᴿ ⁻¹"] = xr.apply_ufunc(pinv, ds["∇ⱼτ̄ᵅ|ᴿ"],
                                   input_core_dims = [["α", "j"]],
                                   output_core_dims = [["j", "α"]],
                                   exclude_dims = set(("α", "j")),
                                   dask_gufunc_kwargs = dict(output_sizes = {'α': len(ds.α), 'j': len(ds.j)}),
                                   dask = "parallelized",
                                   )

    #+++ Test inversion
    if test_propagation:
        print("Test that the inversion propagation is correctly done")
        with ProgressBar():
            A1 = pinv(ds["∇ⱼτ̄ᵅ|ᴿ"].isel(time=0, zC=0, xC=0).data)
            B1 = ds["∇ⱼτ̄ᵅ|ᴿ ⁻¹"].isel(time=0, zC=0, xC=0)
            C1 = ds["∇ⱼτ̄ᵅ|ᴿ"].isel(time=0, zC=0, xC=0)
            assert np.allclose(A1, B1.T), "Something wrong with matrix inversion broadcasting" # Ensure that what we get using `apply_ufunc` == what we get using `pinv` directly
            assert np.allclose((B1.data @ C1.data.T), np.eye(2), atol=1e-6), "Inverse grad matrix isn't correct" # If we indeed computed the pseudo-inverse, B3 @ Cᵀ= identity matrix
            print(Fore.GREEN + f"OK")
            print(Style.RESET_ALL)
    #---

    return ds

def get_transport_tensor(ds, normalize_tracers=True, test_propagation=True):
    """ Returns transport tensor Rᵢⱼ obtained using ⟨uᵢ′τᵅ′⟩ and ∇ⱼτ̄ᵅ
    which should be in dataset `ds`

    `normalize_tracers`: Bool
        Whether or not to normalize tracers by their respective rms.
        `
    `test_propagation`: Bool
        Whether or not to test that the inversion results are propagated correctly
        through the dataset. Catches shape errors in the data.
    """

    #+++ Reshape relevant vectors
    # Following Bachman et al. (2015) the shapes are
    # ⟨uᵢ′τᵅ′⟩   n × m (number of dimensions × number of tracers)
    # ∇ⱼτ̄ᵅ       n × m (number of dimensions × number of tracers)
    # Rᵢⱼ        n × n (number of dimensions × number of dimensions)
    ds["⟨uᵢ′τᵅ′⟩"] = ds["⟨uᵢ′τᵅ′⟩"].transpose(..., "i", "α")
    ds["∇ⱼτ̄ᵅ"]     = ds["∇ⱼτ̄ᵅ"].transpose(..., "j", "α")
    #---

    #+++ Maybe normalize averages before starting the method
    if normalize_tracers:
        rms = np.sqrt((ds["⟨uᵢ′τᵅ′⟩"]**2).mean(("xC", "zC", "i")))

        ds["∇ⱼτ̄ᵅ|ᴿ"]     = ds["∇ⱼτ̄ᵅ"]     / rms
        ds["⟨uᵢ′τᵅ′⟩|ᴿ"] = ds["⟨uᵢ′τᵅ′⟩"] / rms
    #---

    #+++ Get the transport tensor Rᵢⱼ
    # For the matrix multiplication, the shapes are:
    # (n × m) ⋅ (m × n) = (n × n)
    # (number of dimensions × number of tracers) ⋅ (number of tracers × number of dimensions)
    ds = invert_gradient_tensor(ds, test_propagation=test_propagation)
    ds["∇ⱼτ̄ᵅ|ᴿ ⁻¹"] = ds["∇ⱼτ̄ᵅ|ᴿ ⁻¹"].transpose(..., "α", "j")

    ds["Rᵢⱼ"] = -xr.apply_ufunc(np.matmul, ds["⟨uᵢ′τᵅ′⟩|ᴿ"], ds["∇ⱼτ̄ᵅ|ᴿ ⁻¹"],
                                 input_core_dims=[["i", "α"], ["α", "j"]],
                                 output_core_dims=[["i", "j"]],
                                 dask = "parallelized",)

    #+++ Test
    if test_propagation:
        print("Test that matrix multiplication is correctly done")
        with ProgressBar():
            A2 = -ds["⟨uᵢ′τᵅ′⟩|ᴿ"].isel(time=0, zC=0, xC=0) @ ds["∇ⱼτ̄ᵅ|ᴿ ⁻¹"].isel(time=0, zC=0, xC=0)
            B2 =  ds["Rᵢⱼ"].isel(time=0, zC=0, xC=0)
            assert np.allclose(A2, B2), "Something wrong with Rᵢⱼ calculation"
            print(Fore.GREEN + f"OK")
            print(Style.RESET_ALL)
    #---
    #---

    return ds


