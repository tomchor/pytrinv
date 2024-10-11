
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

