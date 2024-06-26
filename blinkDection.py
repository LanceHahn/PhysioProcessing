import stumpy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from matplotlib import colors

from mne.channels import make_standard_montage
from mne.viz import plot_sensors
COLOR_LIST = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
MAX_REAL = 0.01  # threshold value for determining a channel value is invalid

def plotEEGs(eegData, tLabels, eLabels):
    """
    plot a series of eeg electrode traces.
    :param eegData: a list of lists of electrode measurements
    :param tLabels: a list of floating point numbers indicating time
    :param eLabels: list of electrode label strings
    :return:
    """
    base = 0.0
    offset = -0.0003
    for ix in range(len(eLabels)):
        print(f"{eLabels[ix]} ", end="")
        adjusted = [x if x < MAX_REAL else MAX_REAL for x in eegData[ix]]
        if None in adjusted:
            print(f"Unreal (over {MAX_REAL}) values found.")
            print(f"Bad values: {[(ix, val) for ix, val  in enumerate(eegData[ix]) if val >= MAX_REAL]}")
        maxV = min(max(adjusted), MAX_REAL)
        minV = max(min(adjusted), -MAX_REAL)
        adjusted = [x - maxV + base if x < MAX_REAL else None for x in
                    eegData[ix]]
        plt.plot(tLabels, adjusted, label=eLabels[ix])
        base += offset + minV - maxV
    plt.title('initial EEG plot')
    plt.legend()
    plt.show()
    return


def plotMotifDiscovery(timeLabels, vData, distData, targetIX, matchIX, wwidth,
                       title='Motif (Pattern) Discovery'):
    """
    Plot the original wave form with colorful patches highlighting the detected
    pattern and the matrix profile distance values with the lowest distance
    pair highlighted
    :param timeLabels:
    :param vData:
    :param distData:
    :param targetIX:
    :param matchIX:
    :param wwidth:
    :param title:
    :return:
    """
    vMax = max(vData)
    vMin = min(vData)
    height = vMax - vMin
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle(title, fontsize='14')
    axs[0].plot(timeLabels, vData)
    axs[0].set_ylabel('EEG', fontsize='14')
    rect0 = Rectangle((timeLabels[targetIX], vMin), wwidth, height=height-10, facecolor='lightgrey')
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
    plt.suptitle(f'Motif Match (dist: {dist})', fontsize='14')
    sMin = max(0, min(targetIX - (2*wwidth), matchIX - (2*wwidth)))
    sMax = min(len(vData), max(targetIX + (3*wwidth), matchIX + (3*wwidth)))
    axs[0].plot(vData[sMin:sMax])
    axs[0].set_ylabel('EEG', fontsize='14')
    rect0 = Rectangle((targetIX-sMin, vMin), wwidth, height=height-10,
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

def plotMotifMatches(vData, indecies, wwidth, title=None):
    # plot a window that includes the two matched patterns with 2x window size border
    # and the two matched waveforms on top of one another

    vMax = max(vData)
    vMin = min(vData)
    height = vMax - vMin
    fig, axs = plt.subplots(2)
    if title is None:
        title = f'Motif Match ({len(indecies)})'
    plt.suptitle(title, fontsize='14')
    sMin = max(0, min(ix - (2*wwidth) for ix in indecies))
    sMax = min(len(vData), max(ix + (3*wwidth) for ix in indecies))
    axs[0].plot(vData[sMin:sMax], color=COLOR_LIST[0])
    #axs[0].set_ylabel('EEG', fontsize='14')
    for ix, _index in enumerate(indecies):
        rect = Rectangle((_index-sMin, vMin), max(10, wwidth - (5*ix)), height=height,
                          facecolor=COLOR_LIST[1 + (ix % (len(COLOR_LIST)-1))])
        axs[0].add_patch(rect)
    axs[0].tick_params(labelbottom=False)
    axs[1].set_xlabel("Time", fontsize='20')

    for ix, _index in enumerate(indecies):
        axs[1].plot(vData[_index:_index + wwidth], color=COLOR_LIST[1 + (ix % (len(COLOR_LIST)-1))])
    plt.show()
    return

def expandVizWindow(indecies, wwidth, electrodes):
    # Identify common windows for each signal event

    eleCount = len(indecies)
    commonStarts = indecies[0]
    commonWwidth = wwidth
    minOverlap = int(wwidth / 2)
    commonPreWidth = 0
    blinkAdded = False
    binnedCount = [0] * eleCount
    for eIX in range(1, eleCount):  # step through electrodes
        for bIX in indecies[eIX]:  # step through blinks for the electrode
            for comIX in range(len(commonStarts)):
                # step through possible common matches for this blink
                if (commonStarts[comIX] - commonPreWidth <= bIX < commonStarts[comIX] + commonWwidth or
                        commonStarts[comIX] - commonPreWidth <= bIX + wwidth < commonStarts[comIX] + commonWwidth):
                    # this blink for this electrode overlaps with the
                    # current common window. If the common window doesn't
                    # completely engulf the blink then extend either the
                    # beginning or the end (delta)
                    if commonPreWidth + commonWwidth > 1.5 * wwidth:
                        # don't extend window range because the expansion
                        # guardrails suggest this is too extreme
                        blinkAdded = False
                        break
                    # expand the preceding boundary if appropriate
                    if 0 < (commonStarts[comIX] - commonPreWidth) - bIX < minOverlap:
                        commonPreWidth += (commonStarts[comIX] - commonPreWidth) - bIX
                    if 0 < (bIX + wwidth) - (commonStarts[comIX] + commonWwidth) < minOverlap:
                        commonWwidth += (bIX + wwidth) - (commonStarts[comIX] + commonWwidth)
                    blinkAdded = True
                    binnedCount[eIX] += 1
                    break
            if blinkAdded:
                blinkAdded = False
                continue
        print(f"{electrodes[eIX] if electrodes else ' '}{binnedCount[eIX]} of "
              f"{len(indecies[eIX])} ({int(binnedCount[eIX]/len(indecies[eIX])*100)}%) "
              f"binned ({len(commonStarts)} bins)")

    return commonStarts, commonPreWidth, commonWwidth


def plotSynchedMeanWaves(vData, tLabels, indecies, wwidth,
                         electrodes=None, sortSize=True):
    """
    Show the collection of averaged events for each electrode as a
    normalized plot and as a zeroed plot.
    :param vData: list of eeg data for each electrode
    :param tLabels: time labels for eeg data
    :param indecies: index of blink beginning for each electrode
    :param wwidth: blink duration
    :param title: title string
    :param electrodes: electrode labels
    :return: None
    """
    commonStarts, commonPreWidth, commonWwidth = expandVizWindow(indecies, wwidth, electrodes)
    eleCount = len(indecies)

    # Identify common windows for each blink
    # commonStarts = indecies[0]
    # commonWwidth = wwidth
    # minOverlap = int(wwidth / 2)
    # commonPreWidth = 0
    # blinkAdded = False
    # binnedCount = [0] * eleCount
    # for eIX in range(1, eleCount):  # step through electrodes
    #     for bIX in indecies[eIX]:  # step through blinks for the electrode
    #         for comIX in range(len(commonStarts)):
    #             # step through possible common matches for this blink
    #             if (commonStarts[comIX] - commonPreWidth <= bIX < commonStarts[comIX] + commonWwidth or
    #                     commonStarts[comIX] - commonPreWidth <= bIX + wwidth < commonStarts[comIX] + commonWwidth):
    #                 # this blink for this electrode overlaps with the
    #                 # current common window. If the common window doesn't
    #                 # completely engulf the blink then extend either the
    #                 # beginning or the end (delta)
    #                 if commonPreWidth + commonWwidth > 1.5 * wwidth:
    #                     # don't extend window range because the expansion
    #                     # guardrails suggest this is too extreme
    #                     blinkAdded = False
    #                     break
    #                 # expand the preceding boundary if appropriate
    #                 if 0 < (commonStarts[comIX] - commonPreWidth) - bIX < minOverlap:
    #                     commonPreWidth += (commonStarts[comIX] - commonPreWidth) - bIX
    #                 if 0 < (bIX + wwidth) - (commonStarts[comIX] + commonWwidth) < minOverlap:
    #                     commonWwidth += (bIX + wwidth) - (commonStarts[comIX] + commonWwidth)
    #                 blinkAdded = True
    #                 binnedCount[eIX] += 1
    #                 break
    #         if blinkAdded:
    #             blinkAdded = False
    #             continue
    #     print(f"{electrodes[eIX] if electrodes else ' '}{binnedCount[eIX]} of {len(indecies[eIX])} ({int(binnedCount[eIX]/len(indecies[eIX])*100)}%) binned ({len(commonStarts)} bins)")

    # create an average signal for each electrode over the common windows
    synchWaves = []
    window = np.ones(5)
    waveMaxes = []
    timeLabels = tLabels[commonStarts[0] - commonPreWidth:commonStarts[0] + commonWwidth]
    print(f"The {len(commonStarts)} temporal windows used:")
    for comIX in range(len(commonStarts)):
        print(f"{comIX+1}:{tLabels[commonStarts[comIX] - commonPreWidth]}:{tLabels[commonStarts[comIX] + commonWwidth]}")

    print("Electrode, Starting val, Max (t), index, max found by conv, max conv")
    for eIX in range(eleCount):
        synchWaves.append(np.mean([vData[eIX][commonStarts[comIX] - commonPreWidth:commonStarts[comIX] + commonWwidth]
                                  for comIX in range(len(commonStarts))],axis=0))
        ixMax = np.argmax(np.convolve(window, synchWaves[-1], mode='valid'))
        waveMaxes.append((electrodes[eIX], synchWaves[-1][0], timeLabels[ixMax], ixMax, synchWaves[-1][ixMax], np.convolve(window, synchWaves[-1], mode='valid')[ixMax]))
        print(f"{','.join(str(w) for w in waveMaxes[-1])}")

    # Create a list of electrode positions ordered from the largest positive
    # diversion to the largest negative diversion
    pos = [(ix, bb) for ix, bb in enumerate(synchWaves)]
    if sortSize:
        pos.sort(key=lambda x:-(np.max(x[1])-x[1][0]))
    posi = [p[0] for p in pos]

    # Plot sychronized waves with normalized values.
    # plotWaves([synchWaves[p] for p in posi], xLabels=timeLabels, labels=[electrodes[p] for p in posi],
    #           zNorm=True, title=f"Synchronized {len(electrodes)} Electrode Mean ({len(commonStarts)}) Normed Waves")

    # Plot sychronized waves with values shifted vertically so that they all start
    # at the same 0 y-coordinate.
    plotWaves([[v-syn[0] for v in syn] for syn in [synchWaves[p] for p in posi]],
              xLabels=timeLabels, labels=[electrodes[p] for p in posi],
              zNorm=False, title=f"Synchronized {len(electrodes)} Electrode Mean ({len(commonStarts)}) Zeroed Waves")

    return waveMaxes

def plotMotifMatchesMultiElectrodes(vData, tLabels, indecies, wwidth,
                                    title=None, electrodes=None):
    # plot a window that includes the two matched patterns with 2x window size border
    # and the two matched waveforms on top of one another

    eleCount = len(indecies)
    fig, axs = plt.subplots(eleCount)
    if title is None:
        title = f"All {eleCount} electrodes All waves"
    plt.suptitle(title, fontsize='14')

    # find series min and max across all electrodes
    sMin = max(0, min(ix - (2 * wwidth) for ix in indecies[0]))
    sMax = min(len(vData[0]), max(ix + (3 * wwidth) for ix in indecies[0]))
    for eIX in range(1, eleCount):
        if sMin > max(0, min(ix - (2 * wwidth) for ix in indecies[eIX])):
            sMin = max(0, min(ix - (2 * wwidth) for ix in indecies[eIX]))
        if sMax < min(len(vData[eIX]), max(ix + (3 * wwidth) for ix in indecies[eIX])):
            sMax = min(len(vData[eIX]), max(ix + (3 * wwidth) for ix in indecies[eIX]))
    for eIX in range(eleCount):
        vMax = max(vData[eIX])
        vMin = min(vData[eIX])
        height = vMax - vMin
        axs[eIX].plot(tLabels[sMin:sMax], vData[eIX][sMin:sMax], color=COLOR_LIST[0],
                      label=electrodes[eIX])
        for ix, _index in enumerate(indecies[eIX]):
            rect = Rectangle((tLabels[_index-sMin], vMin), max(.1, wwidth/1000), height=height,
                              facecolor=COLOR_LIST[1 + (ix % (len(COLOR_LIST)-1))])
            axs[eIX].add_patch(rect)
        axs[eIX].legend(loc='upper right')
        if eIX > eleCount - 2 or eleCount == 1:
            axs[eIX].tick_params(labelbottom=True)
        else:
            axs[eIX].tick_params(labelbottom=False)
    plt.show()
    return


def plotWaves(waves, xLabels=[], labels=[], zNorm=True, title="Wave Plot"):
    # plot a window that includes the two matched patterns with 2x window size border
    # and the two matched waveforms on top of one another
    if zNorm and title == '':
        title = f"{title} normed"
    for ix, waveO in enumerate(waves):
        if zNorm:
            wave = stumpy.core.z_norm(waveO)
        else:
            wave = waveO
        if len(xLabels) > 0:
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

def findBlinkWave(vData, blinkDuration, sampleHz=1000, tLabels=[],
                  verbose=10, electrode=None):
    """
    Return a wave profile that is a combination of two well-matched waves in the
    sequence.
    :param vData: time series data
    :param blinkDuration: expected blink duration in seconds
    :param sampleHz: the number of samples per second in the data provided
    :param verbose: how verbose (0-10) output should be
    :return: ndarray containing wave profile
    """
    convolve = True
    window_size = int(blinkDuration * sampleHz)  #  data points found in a pattern
    if verbose > 2:
        print(f"Looking across {len(vData)/sampleHz}s sampled at {sampleHz}Hz "
              f"({len(vData)} points) for electrode {electrode} with a window of {blinkDuration}s ({window_size} points)")
    matrix_profile = stumpy.stump(vData, m=window_size)
    mp = matrix_profile
    motif_idx = np.argsort(mp[:, 0])[0]
    if verbose > 2:
        print(f"The motif is located at index {motif_idx}")
    nearest_neighbor_idx = mp[motif_idx, 1]
    closestDistance = mp[motif_idx][0]

    # Using a window of 300 but only the central 200 measurements
    # If this is continued, consider making it simple a mean across 200
    # However it is possible we may want to consider a more complex weighting
    # scheme across the window which could be accomplished with convolution

    if convolve:
        print("Blinks found using convolved window")
        window = np.ones(window_size)
        bbv = np.convolve(window, mp[:, 0], mode='valid')
        if verbose > 8:
            plotWaves([bbv[:]], xLabels=tLabels[:len(bbv)], labels=['conv'],
                      zNorm=False, title=f"Convolution {electrode}")
        bbv_min = np.min(bbv)
        bbmotif_idx = np.where(bbv==bbv_min)[0][0] + 1 + int(window_size/2)
        bbneighbor_idx = mp[bbmotif_idx,1]
        if verbose > 2:
            print(f"The nearest neighbor is located at index {bbneighbor_idx}")
        if verbose > 8:
            plotMotifDiscovery(tLabels, vData, mp[:, 0],
                               bbmotif_idx, bbneighbor_idx,
                               window_size/sampleHz, title=f"windowed Discovery {electrode}")
        if verbose > 7:
            plotMotifMatches(vData, [bbmotif_idx, bbneighbor_idx],
                             window_size, title=f'Motif Match (Electrode {electrode})')
        blinkWave = combineWaves([vData[bbmotif_idx:bbmotif_idx + window_size],
                                 vData[bbneighbor_idx:bbneighbor_idx + window_size]],
                                 [0.5, 0.5])
        if verbose > 8:
            plotWaves([vData[bbmotif_idx:bbmotif_idx + window_size],
                        vData[bbneighbor_idx:bbneighbor_idx + window_size],
                        blinkWave],
                      xLabels=[], zNorm=True,
                      labels=[tLabels[bbmotif_idx], tLabels[bbneighbor_idx], "Blink"],
                      title=f"Convolved (Electrode {electrode}) normed")
    else:  # don't convolve
        if verbose > 2:
            print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")
        if verbose > 8:
            plotMotifDiscovery(tLabels, vData, mp[:, 0],
                               motif_idx, nearest_neighbor_idx,
                               window_size / sampleHz)
        if verbose > 7:
            plotMotifMatches(vData, [motif_idx, nearest_neighbor_idx],
                             window_size, title=f'Motif Match (Electrode {electrode})')

        blinkWave = combineWaves([vData[motif_idx:motif_idx + window_size],
                                  vData[
                                  nearest_neighbor_idx:nearest_neighbor_idx + window_size]],
                                 [0.5, 0.5])
        if verbose > 6:
            plotWaves([vData[motif_idx:motif_idx + window_size],
                       vData[nearest_neighbor_idx:nearest_neighbor_idx + window_size],
                       blinkWave], xLabels=[], zNorm=True,
                      labels=[tLabels[motif_idx],
                              tLabels[nearest_neighbor_idx], "Blink"],
                      title="Waves found normed")
    return blinkWave

def findBlinks(initWave, vData, blinkDuration, sampleHz=1000,
                      tLabels=[], verbose=10, electrode=None):
    """
    Return a list of the start time of a blink
    in seconds and a list of associated wave dissimilarities
    :param vData: time series data
    :param blinkDuration: expected blink duration in seconds
    :param sampleHz: the number of samples per second in the data provided
    :param verbose: how verbose (0-10) output should be
    :return: [blink_start_seconds, ...], [blink dissimilarity, ...]
    """
    window_size = int(blinkDuration * sampleHz)  # data points found in a pattern
    if verbose > 2:
        print(f"Looking across {len(vData) / sampleHz}s sampled at {sampleHz}Hz ({len(vData)} points) with a window of {blinkDuration}s ({window_size} points)")
    distance_profile = stumpy.mass(initWave, vData)
    if verbose > 9:
        plt.plot(tLabels[:len(distance_profile)],
                 distance_profile,
                 label="Dissimilarity from target wave")
        plt.title(f'Distance Profile E {electrode}')
        plt.show()

    idx = int(np.argmin(distance_profile))
    if verbose > 2:
        print(f"The best match to Blink Template is located at index {idx} "
              f"(time: {tLabels[idx]})")
    disProf = np.argsort(distance_profile)
    disThresh = 10 # 5 # 2
    candidateCount = np.argsort(np.where(distance_profile < disThresh)).size
    blinkIxs = [idx]  # blink start times
    blinkDis = [distance_profile[idx]]  # wave dissimilarity from template
    wwidth = len(initWave)
    if verbose > 3:
        print(f"Adding Data Index, Time, Dissimilarity")
    for ix in disProf[:candidateCount]:
        ix_I = int(ix)
        if any(b - wwidth < ix_I < b + wwidth for b in blinkIxs):
            pass
        else:
            if verbose > 3:
                print(f"Adding {ix_I} {tLabels[ix_I]} {distance_profile[ix_I]}")
            blinkIxs.append(ix_I)
            blinkDis.append(distance_profile[ix_I])
    blinkDuo = sorted(zip(blinkIxs, blinkDis), key=lambda x:x[0])
    blinkIxs = [x[0] for x in blinkDuo]
    blinkDis = [x[1] for x in blinkDuo]
    blinks = [tLabels[ix] for ix in blinkIxs]
    if verbose > 3:
        print(f"{len(blinks)} blinks found at {blinks}")

    if verbose > 6:
        plotMotifMatches(vData, blinkIxs, window_size, title=f"{electrode} Motif Matches ({len(blinkIxs)})")
    if verbose > 8:
        plotWaves([initWave]+[vData[bix:bix + window_size] for bix in blinkIxs],
                  xLabels=[],
                  labels=["Blink"] + blinks, title=f"{electrode} Waves Found ({len(blinkIxs)})")
    return blinks, blinkDis, blinkIxs

def  zeroOutOfRange(data):
    minReal = -1 # -0.01
    maxReal = 1 #0.01
    badVal = 0
    cleaned = [x if minReal < x < maxReal else badVal for x in data]
    return cleaned

def stratifyForColors(chVals, ignores, binCount, mn, mx, ignoreVal):
    """
    organize a set of channel values into a collection of ordered bins that
    can be used to specify channel colors for mne.viz.plot_sensors()

    :param chVals: a dictionary of channel labels and associated values
    :param ignores: list of channel labels for channels to ignore
    :param binCount: integer number of levels to separate the values into
    :param mn: minimum value to include
    :param mx: maximum value to include
    :param ignoreVal: value to associate with ignored channels (e.g., mn)
    :return: a list of bins with channel labels organized into them by value
    """
    valueList = [[] for i in range(binCount)]
    rng = mx - mn
    binCnt = binCount - 1
    for chL, chV in chVals.items():
        binIX = int(((chV-mn)*binCnt/rng))
        valueList[binIX].append(chL)
    for chL in ignores:
        binIX = int(((ignoreVal-mn)*binCnt/rng))
        valueList[binIX].append(chL)
    return valueList

def plotSensorStrengths(goodChannels, waveRespMetrics, electLabels, montageFunc, info):

    montage = make_standard_montage('GSN-HydroCel-129')
    montage.ch_names[-1] = 'E129'
    montageFunc(montage, match_case=False)
    ignoreChannels = [x for x in montage.ch_names if x not in goodChannels]
    ignoreIndecies = [electLabels.index(x) for x in ignoreChannels]

    waveRespMetrics.sort(key=lambda x:(x[4] - x[1]))
    print(f"waveRespMetrics: {waveRespMetrics}")
    sensorIndeciesVals = dict({int(w[0][1:])-1: w[4]-w[1] for w in waveRespMetrics})

    cmapf = cm.binary # cm.cool
    resMaxDiff = max(x[4]-x[1] for x in waveRespMetrics)
    resMinDiff = min(x[4]-x[1] for x in waveRespMetrics)
    chCategories = stratifyForColors(sensorIndeciesVals, ignores=ignoreIndecies,
                                     binCount=cmapf.N, mn=resMinDiff, mx=resMaxDiff,
                                     ignoreVal=resMinDiff)

    print("Add colorbar key")
    norm = colors.Normalize(vmin=resMinDiff, vmax=resMaxDiff)
    sm = cm.ScalarMappable(cmap=cmapf, norm=norm)
    fig3, ax = plt.subplots()
    uu = plot_sensors(info, ch_type="eeg", axes=ax, ch_groups=chCategories,
                      cmap=cmapf, linewidth=0, pointsize=50, show_names=False,
                      kind="topomap", to_sphere=True, show=False)
    cbar = uu.colorbar(sm, ax=ax)
    uu.show()