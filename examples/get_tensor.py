import numpy as np
import xarray as xr
from natsort import natsorted
from dask.diagnostics import ProgressBar

import sys
sys.path.append("..")
from pytrinv.utils import condense, separate_tracers, concat_tracers_back
from pytrinv.core import get_transport_tensor, get_Rij_eigenvalues, reconstruct_buoyancy_fluxes, reconstruct_tracer_fluxes
from colorama import Fore, Back, Style

#+++ Options
filename = "xaz.TEST-f32.nc"
indices = [1, 3] # Indices to use when calling `condense()`
inversion_tracers = range(1, 5) # Tracers to use

average_time = False
add_sgs_fluxes = False
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

#+++ Maybe separate tracers into τᵅ and τᵐ
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
# These quantities are the only needed to move forward (with these specific names)
xaz = xaz[["τ̄ᵅ", "∇ⱼτ̄ᵅ",
           "τ̄ᵐ", "∇ⱼτ̄ᵐ",
           "b̄", "∇ⱼb̄",
           "⟨uᵢ′τᵅ′⟩",
           "⟨uᵢ′τᵐ′⟩",
           "⟨uᵢ′b′⟩",
          ]]
#---

# Actual calculation starts here
xaz = get_transport_tensor(xaz, test_propagation=False, normalize_tracers=False)
xaz = get_Rij_eigenvalues(xaz, test_propagation=False)
xaz = reconstruct_buoyancy_fluxes(xaz, test_propagation=False)
xaz = concat_tracers_back(xaz) # Put gradients that weren't used in the inversion back
xaz = reconstruct_tracer_fluxes(xaz, test_propagation=False)
