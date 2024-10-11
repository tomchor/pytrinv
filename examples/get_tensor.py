import numpy as np
import xarray as xr
from natsort import natsorted
import pynanigans as pn
from dask.diagnostics import ProgressBar

import sys
sys.path.append("..")
from pytrinv.utils import condense, separate_tracers, concat_tracers_back
from pytrinv.core import get_transport_tensor
from colorama import Fore, Back, Style

#+++ Options
filename = "xaz.TEST-f32.nc"
test_sel = dict(time=10, xC=1e3, zC=-20, method="nearest")
reduce_dims = ("y",)
indices = [1, 3]
n_inversion_tracers = 4

test_averages = False
test_propagation = False

average_time = False
add_sgs_fluxes = False
normalize_tracers = True
#---

#+++ Open dataset
if __name__ == "__main__": print(f"\nGoing to invert tracers in file {filename}\n")
xaz = xr.load_dataset(filename, decode_times=False).squeeze()
xaz = xaz.rename(b_yavg="b̄") # For consistency with notation
#---

#+++ Merge fluxes and gradients (of tracers and buoyancy) into tensors
#+++ Merge tracer concentrations into one with counter α
xaz = condense(xaz, vlist = natsorted([ v for v in xaz.variables.keys() if (v.startswith("τ") and v.endswith("yavg")) ]), varname="τ̄ᵅ", dimname="α")
#---

#+++ Tracer gradients
xaz = condense(xaz, vlist = natsorted([ v for v in xaz.variables.keys() if (v.startswith("dτ") and v.endswith("dx_yavg")) ]), varname="dτ̄ᵅdx", dimname="α")
xaz = condense(xaz, vlist = natsorted([ v for v in xaz.variables.keys() if (v.startswith("dτ") and v.endswith("dz_yavg")) ]), varname="dτ̄ᵅdz", dimname="α")

xaz = condense(xaz, vlist = ["dτ̄ᵅdx", "dτ̄ᵅdz"], varname="∇ⱼτ̄ᵅ", dimname="j", indices=indices)
#---

#+++ Resolved turbulent tracer fluxes
xaz = condense(xaz, vlist = natsorted([ v for v in xaz.variables.keys() if (v.startswith("u′τ") and v.endswith("_yavg")) ]), varname="⟨u′τᵅ′⟩", dimname="α")
xaz = condense(xaz, vlist = natsorted([ v for v in xaz.variables.keys() if (v.startswith("w′τ") and v.endswith("_yavg")) ]), varname="⟨w′τᵅ′⟩", dimname="α")

xaz = condense(xaz, vlist = ["⟨u′τᵅ′⟩", "⟨w′τᵅ′⟩"], varname="⟨uᵢ′τᵅ′⟩", dimname="i", indices=indices)
#---

#+++ Merge resolved buoyancy fluxes and gradients
xaz = condense(xaz, vlist = ["u′b′_yavg", "w′b′_yavg"], varname="⟨uᵢ′b′⟩", dimname="i", indices=indices)

xaz = condense(xaz, vlist = ["dbdx_yavg", "dbdz_yavg"], varname="∇ⱼb̄", dimname="j", indices=indices)
#---

#+++ SGS tracer and buoyancy fluxes
xaz = condense(xaz, vlist = natsorted([ v for v in xaz.variables.keys() if (v.startswith("qτ") and v.endswith("x_yavg")) ]), varname="⟨qτᵅ_x⟩", dimname="α")
xaz = condense(xaz, vlist = natsorted([ v for v in xaz.variables.keys() if (v.startswith("qτ") and v.endswith("z_yavg")) ]), varname="⟨qτᵅ_z⟩", dimname="α")

xaz = condense(xaz, vlist = ["⟨qτᵅ_x⟩", "⟨qτᵅ_z⟩"], varname="⟨qᵢτᵅ⟩", dimname="i", indices=indices)

xaz = condense(xaz, vlist = ["qb_sgs_x_yavg", "qb_sgs_z_yavg"], varname="⟨qᵢb⟩", dimname="i", indices=indices)
#---
#---

#+++ Maybe separate tracers into τᵅ and tᵅ
if (type(n_inversion_tracers) is float) or (type(n_inversion_tracers) is int):
    if n_inversion_tracers > 0:
        inversion_tracers = xaz.α[:n_inversion_tracers]
    elif n_inversion_tracers < 0:
        inversion_tracers = xaz.α[n_inversion_tracers:]
    else:
        inversion_tracers = xaz.α

if not len(xaz.α) == len(inversion_tracers):
    xaz = separate_tracers(xaz, inversion_tracers=inversion_tracers,
                           used_basename="τ", unused_index="m", unused_index_sup="ᵐ")
#---

#+++ Time averages
if average_time:
    xaz = xaz.mean("time").expand_dims("time").assign_coords(time=[0])
#---

#+++ Add SGS tracer fluxes to resolved ones
if add_sgs_fluxes:
    print("Adding τ SGS fluxes")
    xaz["⟨uᵢ′τᵅ′⟩"] += xaz["⟨qᵢτᵅ⟩"]
    xaz["⟨uᵢ′b′⟩"] += xaz["⟨qᵢb⟩"]
#---

#+++ Clean up dataset
# These quantities are the only needed to move forward
xaz = xaz[["τ̄ᵅ", "∇ⱼτ̄ᵅ",
           "τ̄ᵐ", "∇ⱼτ̄ᵐ",
           "b̄", "∇ⱼb̄",
           "⟨uᵢ′τᵅ′⟩",
           "⟨uᵢ′τᵐ′⟩",
           "⟨uᵢ′b′⟩",
          ]]
#---

xaz = get_transport_tensor(xaz, test_propagation=False)

#+++ Get eigenvalues or Rᵢⱼ
xaz["eig(Rᵢⱼ)"] = xr.apply_ufunc(np.linalg.eigvals, xaz["Rᵢⱼ"],
                                 input_core_dims=[["i", "j"]],
                                 output_core_dims=[["i"]],
                                 dask = "parallelized",)
xaz["eig(Rᵢⱼ)"] = xaz["eig(Rᵢⱼ)"].rename(i="k")

#+++ Test
if test_propagation:
    print("Test that eigenvalue calculation is correctly propagated")
    with ProgressBar():
        A3 = np.linalg.eigvals(xaz["Rᵢⱼ"].sel(**test_sel))
        B3 = xaz["eig(Rᵢⱼ)"].sel(**test_sel).values
        assert np.allclose(A3, B3), "Eigenvalue calculation isn't correct"
        print(Fore.GREEN + f"OK")
        print(Style.RESET_ALL)
#---
#---

#+++ Reconstruct buoyancy fluxes
xaz["∇ⱼb̄"] = xaz["∇ⱼb̄"].expand_dims(dim=dict(μ=[1])) # Extra dimension is needed by apply_ufunc, for some reason
xaz["∇ⱼb̄"] = xaz["∇ⱼb̄"].transpose(..., "j", "μ")
xaz["⟨uᵢ′b′⟩ᵣ"] = -xr.apply_ufunc(np.matmul, xaz["Rᵢⱼ"], xaz["∇ⱼb̄"],
                                  input_core_dims = [["i", "j"], ["j", "μ"]],
                                  output_core_dims = [["i", "μ"]],
                                  dask = "parallelized",)

xaz = xaz.squeeze("μ").drop_vars("μ") # Get rid of μ which is no longer needed

#+++ Test
if test_propagation:
    print("Test that buoyancy flux reconstruction is done correctly")
    with ProgressBar():
        A4 = -xaz["Rᵢⱼ"].isel(time=0, zC=0, xC=0).values @ xaz["∇ⱼb̄"].isel(time=0, zC=0, xC=0).values
        B4 =  xaz["⟨uᵢ′b′⟩ᵣ"].isel(time=0, zC=0, xC=0)
        assert np.allclose(A4, B4.values), "Buoyancy flux reconstruction isn't correct"
        print(Fore.GREEN + f"OK")
        print(Style.RESET_ALL)
#---
#---

#+++ Reconstruct passive tracer fluxes
if n_inversion_tracers: # Put gradients that weren't used in the inversion back
    xaz = concat_tracers_back(xaz)

xaz["∇ⱼτ̄ᵅ"] = xaz["∇ⱼτ̄ᵅ"].expand_dims(dim=dict(μ=[1])) # Extra dimension is needed by apply_ufunc, for some reason
xaz["∇ⱼτ̄ᵅ"] = xaz["∇ⱼτ̄ᵅ"].transpose(..., "j", "μ")
xaz["⟨uᵢ′τᵅ′⟩ᵣ"] = -xr.apply_ufunc(np.matmul, xaz["Rᵢⱼ"], xaz["∇ⱼτ̄ᵅ"],
                                   input_core_dims = [["i", "j"], ["j", "μ"]],
                                   output_core_dims = [["i", "μ"]],
                                   dask = "parallelized",)

xaz = xaz.squeeze("μ").drop_vars("μ") # Get rid of μ which is no longer needed

#+++ Test
if test_propagation:
    print("Test that passive tracer flux reconstruction is done correctly")
    with ProgressBar():
        A5 = -xaz["Rᵢⱼ"].isel(time=0, zC=0, xC=0).values @ xaz["∇ⱼτ̄ᵅ"].isel(time=0, zC=0, xC=0).values.T
        B5 =  xaz["⟨uᵢ′τᵅ′⟩ᵣ"].isel(time=0, zC=0, xC=0)
        assert np.allclose(A5.T, B5.values), "Buoyancy flux reconstruction isn't correct"
        print(Fore.GREEN + f"OK")
        print(Style.RESET_ALL)
#---
#---
