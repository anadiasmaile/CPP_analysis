import numpy as np

def coordinates(evoked,time1, time2):

    """find the two coordinates given two timepoints"""
    idx1 = next(x for x, val in enumerate(evoked.times) if val > time1)
    coord_1 = [evoked.times[idx1], np.sqrt(evoked.data[:, idx1] ** 2)
              .mean(axis = 0)]
    idx2 = next(x for x, val in enumerate(evoked.times) if val > time2)
    coord_2 = [evoked.times[idx2],np.sqrt(evoked.data[:, idx2] ** 2)
              .mean(axis = 0)]
    return coord_1, coord_2

def find_peak(evoked, mode, start, end):

    """find peak within a given time-interval"""
    if mode == 'stim':
        signal = np.zeros((3001,2))
    else:
        signal = np.zeros((3201,2))
    signal[:,0] = evoked.times
    idxs = next(x for x, val in enumerate(evoked.times) if val > start)
    idxe = next(x for x, val in enumerate(evoked.times) if val > end)
    for i in range(len(evoked.times)):
        signal[i,1] = np.sqrt(evoked.data[:, i] ** 2).mean(axis = 0)
    signal = signal[idxs:idxe+1]
    peak = max(signal[:,1])
    return peak

def slope_calc(x1, y1, x2, y2):

    """calculate the slope given two coordinates"""
    s = (y2-y1)/(x2-x1)
    return s
