import numpy as np
import pandas as pd
import math
import pprint

def aggregate_xy(xy_list, method="average", average_resolution=500):
    """ Aggregate multiple xy arrays producing an y average including std-error.

    Parameters
    ----------
    xy_list: 2d-array
        list of `[x,y]` arrays (x MUST be monotonically increasing !)
    method: str
        only 'average' method supported
    average_resolution: int
        number of interpolation points (x-axis)
    
    Returns
    -------
    2d-array
        Interpolated arrays of *precision*, *recall*, *stderr*.
    """
    #pdb.set_trace()
    #print(pprint.pprint(xy_list, width=200))
    if xy_list:
        # Filtering data with missing value
        is_valid = lambda dc: dc[0].size != 0 and dc[1].size != 0 and np.all(~np.isnan(dc[0])) and np.all(~np.isnan(dc[1]))        
        xy_list_filtered = [dc for dc in xy_list if is_valid(dc)]
        if xy_list_filtered:
            # Longest x axis
            max_fa_list = [max(dc[0]) for dc in xy_list_filtered]
            max_fa = max(max_fa_list)
            if method == "average":
                x = np.linspace(0, max_fa, average_resolution)
                ys = np.vstack([np.interp(x, data[0], data[1]) for data in xy_list_filtered])                
                stds = np.std(ys, axis=0, ddof=0)
                n = len(ys)
                stds = stds / math.sqrt(n)
                stds = 1.96 * stds
                # (0,1) (minpfa, 1)
                ys = [np.interp(x,
                                np.concatenate((np.array([0, data[0].min()]), data[0])),
                                np.concatenate((np.array([1, 1]),             data[1])))
                                for data in xy_list_filtered]
                aggregated_dc = [ x, (np.vstack(ys).sum(0) + len(xy_list) - len(xy_list_filtered)) / len(xy_list), stds ]
                return aggregated_dc
    log.error("Warning: No data remained after filtering, returning an empty array list")
    return [ [], [], [] ]

