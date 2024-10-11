import numpy as np
import xarray as xr

def condense(ds, vlist, varname, dimname="α", indices=None):
    """
    Condense variables in `vlist` into one variable named `varname`.
    In the process, individual variables in `vlist` are removed from `ds`.
    """
    if indices == None:
        indices = range(1, len(vlist)+1)

    ds[varname] = ds[vlist].to_array(dim=dimname).assign_coords({dimname : indices})
    ds = ds.drop(vlist)
    return ds


def separate_tracers(ds, used_basename="τ", inversion_tracers=np.arange(1, 5), unused_basename="τ", unused_index="m", unused_index_sup="ᵐ"):
    """
    Separate tracers into ones used in the inversion process and ones unused in the inversion process
    """
    #+++ Test if `inversion_tracers` makes sense
    inversion_tracers = sorted(inversion_tracers)
    inversion_tracers_contained_in_α = [ inversion_tracer in list(ds.α) for inversion_tracer in inversion_tracers ]
    if all(inversion_tracers_contained_in_α):
        if (len(ds.α) == len(inversion_tracers)):
            if all(natsorted(ds.α) == natsorted(inversion_tracers)): # If we use all tracers for the reconstruction there's nothing to be done
                return ds
            else:
                raise(ValueError(f"Something's wrong. Maybe repeated values for `inversion_tracers`?"))
    else:
        raise(ValueError(f"`inversion_tracers` = {inversion_tracers} is not contained in α = {ds.α.values}."))
    #---

    #+++ Do what we're here for
    for var in ds.variables:
        if ("α" in ds[var].coords) and (var != "α"):
            ds["α_in_inversion_tracers"] = xr.DataArray([ α_val in list(inversion_tracers) for α_val in ds[var].α ], dims="α")
            newvar = var.replace(used_basename, unused_basename).replace("ᵅ", unused_index_sup)
            ds[newvar] = ds[var].where(np.logical_not(ds.α_in_inversion_tracers), drop=True).rename(α=unused_index)
    ds = ds.sel(α=inversion_tracers)
    return ds
    #---


def concat_tracers_back(ds, used_basename="τ", unused_basename="τ", unused_index="m", unused_index_sup="ᵐ"):
    """
    Merge back tracers that were separated into ones used in the inversion process and ones unused in the inversion process
    """
    if "m" not in ds: # If there's no m is empty that means we used all tracers for the reconstruction and there's nothing to be done
        return ds

    α_original = sorted(np.concatenate([ds.α, ds[unused_index]]))
    ds_aux = ds.reindex(α=α_original)
    for var in ds.variables:
        if (unused_index in ds[var].coords) and (var != unused_index):
            original_var = var.replace(unused_basename, used_basename).replace(unused_index_sup, "ᵅ")
            ds_aux[original_var] = xr.concat([ds[original_var], ds[var].rename({unused_index : "α"})], dim="α").sortby("α")
    return ds_aux

