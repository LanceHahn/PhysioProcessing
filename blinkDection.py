import stumpy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def findBlinks(vData, blinkDuration, interBlink, sampleHz=1000):
    """
    Return a list of tuples which define the start and end time of a blink
    in seconds
    :param vData: time series data
    :param blinkDuration: expected blink duration in seconds
    :param interBlink:  expected nonblinking interval between blinks
    :param sampleHz: the number of samples per second in the data provided
    :return: [(blink_start_seconds, blink_end_seconds), (), ...]
    """
    blinks = []
    vMax = max(vData)
    vMin = min(vData)
    height = vMax - vMin
    window_size = int(blinkDuration * sampleHz)  #  data points found in a pattern
    print(f"Looking across {len(vData)/sampleHz}s sampled at {sampleHz}Hz ({len(vData)} points) with a window of {blinkDuration}s ({window_size} points)")
    matrix_profile = stumpy.stump(vData, m=window_size)
    mp = matrix_profile
    motif_idx = np.argsort(mp[:, 0])[0]
    print(f"The motif is located at index {motif_idx}")

    nearest_neighbor_idx = mp[motif_idx, 1]
    print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})

    # Plot the original wave form with colorful patches highlighting the detected pattern
    # and the matrix profile distance values with the lowest distance pair highlighted
    plt.suptitle('Motif (Pattern) Discovery', fontsize='14')
    axs[0].plot(vData)
    axs[0].set_ylabel('EEG', fontsize='14')
    rect0 = Rectangle((motif_idx, vMin), window_size, height=height, facecolor='lightgrey')
    axs[0].add_patch(rect0)
    rect1 = Rectangle((nearest_neighbor_idx, vMin), window_size, height=height, facecolor='orange')
    axs[0].add_patch(rect1)
    axs[1].set_xlabel('Time', fontsize='14')
    axs[1].set_ylabel('Matrix Profile', fontsize='14')
    axs[1].axvline(x=motif_idx, linestyle="dashed")
    axs[1].axvline(x=nearest_neighbor_idx, linestyle="dotted")
    axs[1].plot(mp[:, 0])
    plt.show()

    # plot a window that includes the two matched patterns with 2x window size border
    # and the two matched waveforms on top of one another
    fig, axs = plt.subplots(2)
    plt.suptitle('Motif (Pattern) Discovery', fontsize='14')
    sMin = max(0, min(motif_idx - (2*window_size), nearest_neighbor_idx - (2*window_size)))
    sMax = min(len(vData), max(motif_idx + (3*window_size), nearest_neighbor_idx + (3*window_size)))
    axs[0].plot(vData[sMin:sMax])
    axs[0].set_ylabel('EEG', fontsize='14')
    rect0 = Rectangle((motif_idx-sMin, vMin), window_size, height=height, facecolor='palegreen')
    axs[0].add_patch(rect0)
    rect1 = Rectangle((nearest_neighbor_idx-sMin, vMin), window_size, height=height, facecolor='orange')
    axs[0].add_patch(rect1)

    axs[1].set_xlabel("Time", fontsize='20')
    axs[1].set_ylabel("Motif", fontsize='20')
    axs[1].plot(vData[motif_idx:motif_idx + window_size], color='palegreen')
    axs[1].plot(vData[nearest_neighbor_idx:nearest_neighbor_idx + window_size], color='orange')
    plt.show()

    print(f"z-normalized Euclidean between the two waveforms: {mp[motif_idx, 0]}")
    eucSort = np.sort(mp[:, 0])
    plt.plot(eucSort)
    plt.title('Euclidean distances sorted')
    plt.legend()
    plt.show()

    return blinks