import stumpy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plotMotifDiscovery(timeLabels, vData, distData, targetIX, matchIX, wwidth):

    # Plot the original wave form with colorful patches highlighting the detected pattern
    # and the matrix profile distance values with the lowest distance pair highlighted
    vMax = max(vData)
    vMin = min(vData)
    height = vMax - vMin
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='14')
    axs[0].plot(timeLabels, vData)
    axs[0].set_ylabel('EEG', fontsize='14')
    rect0 = Rectangle((timeLabels[targetIX], vMin), wwidth, height=height, facecolor='lightgrey')
    axs[0].add_patch(rect0)
    rect1 = Rectangle((timeLabels[matchIX], vMin), wwidth, height=height, facecolor='orange')
    axs[0].add_patch(rect1)
    axs[1].set_xlabel('Time', fontsize='14')
    axs[1].set_ylabel('Matrix Profile', fontsize='14')
    axs[1].axvline(x=timeLabels[targetIX], linestyle="dashed")
    axs[1].axvline(x=timeLabels[matchIX], linestyle="dotted")
    axs[1].tick_params(labelbottom=True)
    axs[1].plot(timeLabels[:len(distData)], distData)
    plt.show()

    return

def plotMotifMatch(vData, targetIX, matchIX, wwidth, dist):
    # plot a window that includes the two matched patterns with 2x window size border
    # and the two matched waveforms on top of one another
    vMax = max(vData)
    vMin = min(vData)
    height = vMax - vMin
    fig, axs = plt.subplots(2)
    plt.suptitle(f'Motix Match (dist: {dist})', fontsize='14')
    sMin = max(0, min(targetIX - (2*wwidth), matchIX - (2*wwidth)))
    sMax = min(len(vData), max(targetIX + (3*wwidth), matchIX + (3*wwidth)))
    axs[0].plot(vData[sMin:sMax])
    axs[0].set_ylabel('EEG', fontsize='14')
    rect0 = Rectangle((targetIX-sMin, vMin), wwidth, height=height,
                      facecolor='palegreen')
    axs[0].add_patch(rect0)
    rect1 = Rectangle((matchIX-sMin, vMin), wwidth, height=height,
                      facecolor='orange')
    axs[0].tick_params(labelbottom=False)
    axs[0].add_patch(rect1)

    axs[1].set_xlabel("Time", fontsize='20')
    axs[1].set_ylabel("Motif", fontsize='20')
    axs[1].plot(vData[targetIX:targetIX + wwidth], color='palegreen')
    axs[1].plot(vData[matchIX:matchIX + wwidth], color='orange')
    plt.show()
    return


def plotWaves(waves, xLabels=[], labels=[], zNorm=True, title="Wave Plot"):
    # plot a window that includes the two matched patterns with 2x window size border
    # and the two matched waveforms on top of one another
    for ix, waveO in enumerate(waves):
        if zNorm:
            wave = stumpy.core.z_norm(waveO)
        else:
            wave = waveO
        if xLabels:
            plt.plot(xLabels, wave,
                     label=labels[ix])
        else:
            plt.plot(wave,
                     label=labels[ix])
    plt.title(title)
    plt.legend()
    plt.show()
    return

def combineWaves(waves, weights=[]):
    if weights == []:
        weights = [1/len(waves)]* len(waves)
    waveProduct = np.sum([waves[ix] * weights[ix] for ix in range(len(waves))], axis=0)
    return waveProduct

def findBlinkWave(vData, blinkDuration, interBlink, sampleHz=1000, tLabels=[],
                  verbose=10):
    """
    Return a list of tuples which define the start and end time of a blink
    in seconds
    :param vData: time series data
    :param blinkDuration: expected blink duration in seconds
    :param interBlink:  expected nonblinking interval between blinks
    :param sampleHz: the number of samples per second in the data provided
    :return: [(blink_start_seconds, blink_end_seconds), (), ...]
    """
    window_size = int(blinkDuration * sampleHz)  #  data points found in a pattern
    if verbose > 2:
        print(f"Looking across {len(vData)/sampleHz}s sampled at {sampleHz}Hz "
              f"({len(vData)} points) with a window of {blinkDuration}s ({window_size} points)")
    matrix_profile = stumpy.stump(vData, m=window_size)
    mp = matrix_profile
    motif_idx = np.argsort(mp[:, 0])[0]
    if verbose > 2:
        print(f"The motif is located at index {motif_idx}")
    nearest_neighbor_idx = mp[motif_idx, 1]
    closestDistance = mp[motif_idx][0]

    if verbose > 2:
        print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")
    if verbose > 8:
        plotMotifDiscovery(tLabels, vData, mp[:, 0],
                           motif_idx, nearest_neighbor_idx,
                           window_size/sampleHz)
    if verbose > 7:
        plotMotifMatch(vData, motif_idx, nearest_neighbor_idx, window_size,
                       closestDistance)

    blinkWave = combineWaves([vData[motif_idx:motif_idx + window_size],
                             vData[nearest_neighbor_idx:nearest_neighbor_idx + window_size]],
                             [0.5, 0.5])
    if verbose > 6:
        plotWaves([vData[motif_idx:motif_idx + window_size],
                    vData[nearest_neighbor_idx:nearest_neighbor_idx + window_size],
                    blinkWave],
                  xLabels=[],
                  labels=[tLabels[motif_idx], tLabels[nearest_neighbor_idx], "Blink"])

    return blinkWave

def findBlinks(initWave, vData, blinkDuration, sampleHz=1000,
                      tLabels=[], verbose=10):
    """
    Return a list of tuples which define the start and end time of a blink
    in seconds
    :param vData: time series data
    :param blinkDuration: expected blink duration in seconds
    :param sampleHz: the number of samples per second in the data provided
    :return: [(blink_start_seconds, confidence), (), ...]
    """
    window_size = int(blinkDuration * sampleHz)  # data points found in a pattern
    if verbose > 2:
        print(f"Looking across {len(vData) / sampleHz}s sampled at {sampleHz}Hz ({len(vData)} points) with a window of {blinkDuration}s ({window_size} points)")
    distance_profile = stumpy.mass(initWave, vData)
    if verbose > 8:
        plt.plot(tLabels[:len(distance_profile)],
                 distance_profile,
                 label="Dissimilarity from target wave")
        plt.title('Distance Profile')
        plt.show()

    idx = np.argmin(distance_profile)
    if verbose > 2:
        print(f"The best match to Blink Template is located at index {idx} "
              f"(time: {tLabels[idx]})")
    disProf = np.argsort(distance_profile)
    disThresh = 2
    candidateCount = np.argsort(np.where(distance_profile < disThresh)).size
    blinkIxs = [idx]  # blink start times
    wwidth = len(initWave)
    for ix in disProf[:candidateCount]:
        if any(b - wwidth < ix < b + wwidth for b in blinkIxs):
            pass
        else:
            if verbose > 3:
                print(f"Adding {ix} {tLabels[ix]} {distance_profile[ix]}")
            blinkIxs.append(ix)
    blinks = [tLabels[ix] for ix in blinkIxs]
    if verbose > 3:
        print(f"{len(blinks)} blinks found at {blinks}")

    if verbose > 6:
        plotWaves([initWave]+[vData[bix:bix + window_size] for bix in blinkIxs],
                  xLabels=[],
                  labels=["Blink"] + blinks, title=f"Waves Found ({len(blinkIxs)})")
    return blinks