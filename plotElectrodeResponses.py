import copy
import sys
import argparse
import numpy as np
from mne.io.eeglab import read_raw_eeglab, read_epochs_eeglab
from blinkDection import (findBlinkWave, findBlinks, combineWaves, plotWaves, zeroOutOfRange,
                          plotMotifMatchesMultiElectrodes, plotEEGs, plotMotifMatches,
                          plotSynchedMeanWaves)

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


def parse_args(params) -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--interactive', type=str, default='YES')
    parser.add_argument('--dataFile', type=str, default='data/raw/ACL_035_raw.set')
    parser.add_argument('--sampleRate', type=int, default=1000)
    parser.add_argument('--eventDuration', type=int, default=300)
    parser.add_argument('--channels', type=str, default='14 8 1 All')
    parser.add_argument('--badChannels', type=str, default='44')
    parser.add_argument('--learnStart', type=int, default=707)
    parser.add_argument('--learnStop', type=int, default=721)
    parser.add_argument('--findStart', type=int, default=900)
    parser.add_argument('--findStop', type=int, default=1000)

    args = parser.parse_args(params)
    return args

def main(params):

    args = parse_args(params)
    fnameSetRaw = args.dataFile
    askUser = args.interactive == 'YES'
    sampleRate = args.sampleRate
    blinkDurationMS = args.eventDuration
    channelString = args.channels
    badChannelString = args.badChannels
    learnStartTime = args.learnStart
    learnStopTime = args.learnStop
    findStartTime = args.findStart
    findStopTime = args.findStop

    #fnameSetRaw = 'data/raw/ACL_035_raw.set'
    testRaw = read_raw_eeglab(input_fname=fnameSetRaw)

    allData = testRaw.get_data()
    tLabels = testRaw.times
    electLabels = testRaw.ch_names
    print(f"{len(electLabels)} Electrode labels found: {electLabels}")
    print(f"{len(tLabels)} Time labels found: {tLabels}")
    base = 0.0
    maxReal = 0.01
    if askUser:
        channelString = input("Provide a space-delimited list of channel numbers (defaulty 14): ")
        if len(channelString) < 1:
            channelString = '14'
    AllElect = False
    if 'ALL' in channelString.upper():
        AllElect = True
        goodChannels = channelString.split(' ')
        goodChannels = ["E" + ch for ch in goodChannels if ch.upper() != 'ALL']
        if askUser:
            badChannelString = input("Provide a space-delimited list of BAD channel numbers (default 44): ")
            if len(badChannelString) < 1:
                badChannelString = '44'
        badChannels = ["E" + ch for ch in badChannelString.split(' ')]
        goodChannels.extend([e for e in electLabels if e not in goodChannels and e not in badChannels])
    else:
        goodChannels = channelString.split(' ')
        goodChannels = ["E" + ch for ch in goodChannels]
    goodIndecies = [electLabels.index(x) for x in goodChannels]
    #sampleRate = 1000
    #blinkDurationMS = 300  # 150
    waveDuration = blinkDurationMS
    extend = False
    goBig = True
    delta = 30
    if not extend:
        print(f"extend is hard-coded to be False which means that the wave will be held to the {blinkDurationMS}ms expected time window.")
        print(f"Otherwise, the initial time window would be extended at {delta} intervals to find the widest temporal window.")
    print(f"Sample Rate: {sampleRate}    extend: {extend}    ")
    if askUser:
        timeString = input("Provide a start time in seconds (default: 707):")
        if len(timeString) < 1:
            timeString = '707'
        startTime = int(timeString)
    else:
        startTime = learnStartTime
    if askUser:
        timeString = input("Provide a end time in seconds (default: 721):")
        if len(timeString) < 1:
            timeString = '721'
        endTime = int(timeString)
    else:
        endTime = learnStopTime
    print("Generating plot of electrode signal(s)")
    startIX = startTime * sampleRate
    endIX = endTime * sampleRate
    if not AllElect:
        plotEEGs([allData[electIX][startIX:endIX] for electIX in goodIndecies],
                 tLabels[startIX:endIX],
                 [electLabels[electIX] for electIX in goodIndecies])

    blinkOutcomes = {}
    for electIX in goodIndecies:
        print(f"\n*** Processing electrode {electLabels[electIX]}")
        blinkDuration = blinkDurationMS / sampleRate
        blinkOutcomes[electIX] = {
            'original': {},
            'extended': {},
            'Big': {}
        }
        sequ = allData[electIX][startTime * sampleRate:endTime * sampleRate]
        # Find initial blink wave as best duplicated sequence.
        blinkWave = findBlinkWave(sequ, blinkDuration,
                                  sampleHz=sampleRate,
                                  tLabels=tLabels[startTime*1000:endTime*1000],
                                  verbose=2 if not AllElect else 0, electrode=electLabels[electIX])

        blinks, blinkDis, _ = findBlinks(blinkWave, sequ, blinkDuration,
                            sampleHz=sampleRate,
                            tLabels=tLabels[startTime*1000:endTime*1000],
                            verbose=3 if not AllElect else 0, electrode=electLabels[electIX])
        startIndecies = [np.where(tLabels == b)[0][0] - (startTime*1000) for b in blinks]
        newBlinkWave = combineWaves([sequ[start: start + blinkDurationMS]
                                     for start in startIndecies])
        print(f"Blinks per minute: {len(blinks)/((endTime-startTime)/60)}")

        # Try again with new updated wave
        print(f"Repeat blink discover with wave generated from {len(blinks)} "
              f"detected waves.")

        blinks1, blinksDis1, blinkIXs1 = findBlinks(newBlinkWave, sequ, blinkDuration,
                                         sampleHz=sampleRate,
                                         tLabels=tLabels[startTime*1000:endTime*1000],
                                         verbose=4 if not AllElect else 0, electrode=electLabels[electIX])
        blinkOutcomes[electIX]['original']['blinkWave'] = copy.deepcopy(newBlinkWave)
        blinkOutcomes[electIX]['original']['blinks'] = blinks1
        blinkOutcomes[electIX]['original']['blinksIndecies'] = blinkIXs1
        blinkOutcomes[electIX]['original']['dissimilarity'] = blinksDis1
        blinkOutcomes[electIX]['original']['duration'] = blinkDurationMS
        print(f"{len(blinks1)} Blinks per minute: {len(blinks1)/((endTime-startTime)/60)}")
        startIndecies = [np.where(tLabels == b)[0][0] - (startTime * 1000) for
                         b in blinks1]
        if extend:
            # EXTEND window until it alters the number of blinks discovered
            print(f"extending time window from {blinkDurationMS} by steps of {delta}ms until it alters the number of blinks...")
            expectedBlinks = len(blinks1)
            blinkCount = expectedBlinks
            blinkOutcomes[electIX]['extended']['blinks'] =  copy.deepcopy(blinkOutcomes[electIX]['original']['blinks'])
            blinkOutcomes[electIX]['extended']['blinksIndecies'] =  copy.deepcopy(blinkOutcomes[electIX]['original']['blinksIndecies'])
            blinkOutcomes[electIX]['extended']['dissimilarity'] = copy.deepcopy(blinkOutcomes[electIX]['original']['dissimilarity'])
            blinkOutcomes[electIX]['extended']['blinkWave'] = copy.deepcopy(newBlinkWave)
            while blinkCount == expectedBlinks:
                waveDuration += delta
                blinkDuration = waveDuration / sampleRate  # unit: seconds
                print(f"Considering wave duration of {waveDuration}...")
                sequ = allData[electIX][startTime * sampleRate:endTime * sampleRate]

                # Find initial blink wave as best duplicated sequence.
                blinkWaveW0 = findBlinkWave(sequ, blinkDuration,
                                          sampleHz=sampleRate,
                                          tLabels=tLabels[startTime * 1000:endTime * 1000],
                                          verbose=0, electrode=electLabels[electIX])

                blinksW0, blinkDisW0, _ = findBlinks(blinkWaveW0, sequ, blinkDuration,
                                              sampleHz=sampleRate,
                                              tLabels=tLabels[startTime * 1000:endTime * 1000],
                                              verbose=0, electrode=electLabels[electIX])

                startIndecies = [np.where(tLabels == b)[0][0] - (startTime * 1000) for
                                 b in blinksW0]
                newBlinkWave = combineWaves([sequ[start: start + waveDuration]
                                             for start in startIndecies])

                print(f"{len(blinksW0)} Blinks per minute: {len(blinksW0)/((endTime-startTime)/60)}")

                # Try again with new updated wave
                print(f"Repeat blink discovery with wave generated from {len(blinksW0)} "
                      f"detected waves.")
                blinksW2, blinkDisW2, blinksIXsW2 = findBlinks(newBlinkWave, sequ, blinkDuration,
                                                sampleHz=sampleRate,
                                                tLabels=tLabels[startTime * 1000: endTime * 1000],
                                                verbose=3 if not AllElect else 0, electrode=electLabels[electIX])
                print(f"{len(blinksW2)} Blinks per minute: {len(blinksW2)/((endTime-startTime)/60)}")
                blinkCount = len(blinksW2)
                if blinkCount == expectedBlinks:
                    blinkOutcomes[electIX]['extended']['blinks'] = copy.deepcopy(blinksW2)
                    blinkOutcomes[electIX]['extended']['blinksIndecies'] = copy.deepcopy(blinksIXsW2)
                    blinkOutcomes[electIX]['extended']['dissimilarity'] = copy.deepcopy(blinkDisW2)
                    blinkOutcomes[electIX]['extended']['blinkWave'] = copy.deepcopy(newBlinkWave)
                    blinkOutcomes[electIX]['extended']['duration'] = waveDuration
            waveDuration -= delta
            if not AllElect:
                plotWaves([blinkOutcomes[electIX]['extended']['blinkWave']], xLabels=[],
                          labels=[f'blink on {goodChannels[goodIndecies.index(electIX)]}'],
                          title=f"Extended wave {waveDuration} with {len(blinkOutcomes[electIX]['extended']['blinks'])} blinks"
                                f" on {goodChannels[goodIndecies.index(electIX)]}")
        else:
            if not AllElect:
                plotWaves([blinkOutcomes[electIX]['original']['blinkWave']], xLabels=[],
                          labels=[f'blink on {goodChannels[goodIndecies.index(electIX)]}'],
                          title=f"Final wave {waveDuration} with {len(blinkOutcomes[electIX]['original']['blinks'])} blinks"
                                f" on {goodChannels[goodIndecies.index(electIX)]}")
    if extend:
        plotWaves([blinkOutcomes[ix]['extended']['blinkWave'] for ix in goodIndecies], xLabels=[],
                  labels=goodChannels,
                  title=f"Final wave(s) {', '.join([str(blinkOutcomes[k]['extended']['duration']) for k in blinkOutcomes.keys()])}ms with "
                        f"{', '.join([str(len(blinkOutcomes[k]['extended']['blinks'])) for k in blinkOutcomes.keys()])} blinks"
                        f" on {', '.join(goodChannels)}")
    else:
        plotWaves([blinkOutcomes[ix]['original']['blinkWave'] for ix in goodIndecies],
                  xLabels=[], labels=goodChannels,
                  title=f"Final wave(s) {blinkDurationMS}ms with "
                        f"{', '.join([str(len(blinkOutcomes[k]['original']['blinks'])) for k in blinkOutcomes.keys()])} blinks"
                        f" on {', '.join(goodChannels)}")
        startIX = startTime * sampleRate
        endIX = endTime * sampleRate
        if len(goodIndecies) == 1:
            electIX = goodIndecies[0]
            plotMotifMatches(allData[electIX][startIX:endIX], blinkOutcomes[electIX]['original']['blinksIndecies'],
                             waveDuration,
                             title=f'Motif Match (Electrode {electLabels[electIX]})')
        else:
            if not AllElect:
                eeg = []
                for electIX in goodIndecies:
                    eeg.append(allData[electIX][startIX:endIX])
                patches = [blinkOutcomes[electIX]['original']['blinksIndecies'] for electIX in goodIndecies]
                AllBlinks = [electLabels[electIX] for electIX in goodIndecies]
                plotMotifMatchesMultiElectrodes(eeg, tLabels[startTime*1000:endTime*1000], patches, waveDuration,
                                                title="All electrodes All waves '", electrodes=AllBlinks)
        for electIX in goodIndecies:
            print(f"{electLabels[electIX]} ({len(blinkOutcomes[electIX]['original']['blinks'])}): "
                  f"{blinkOutcomes[electIX]['original']['blinks']}")

    plotSynchedMeanWaves([allData[electIX][startIX:endIX] for electIX in goodIndecies],
                         tLabels[startIX:endIX],
                         [blinkOutcomes[electIX]['original']['blinksIndecies'] for electIX in goodIndecies],
                         waveDuration,
                         electrodes=[electLabels[electIX] for electIX in goodIndecies])
    if goBig:
        ### apply wave detection to full range of data
        print("Going Big (longer timeline)")
        print(f"Data time range is from 0 to {int(len(tLabels)/sampleRate)} seconds")
        if askUser:
            timeString = input("Provide a start time in seconds (default: 707):")
            if len(timeString) < 1:
                timeString = '707'
            startTime = int(timeString)
        else:
            startTime = findStartTime
        if askUser:
            timeString = input("Provide a end time in seconds (default: 721):")
            if len(timeString) < 1:
                timeString = '721'
            endTime = int(timeString)
        else:
            endTime = findStopTime

        print("Generating plot of electrode signal(s)")
        startIX = startTime * sampleRate
        endIX = endTime * sampleRate
        cleanData = [zeroOutOfRange(allData[electIX][startIX:endIX]) for electIX in goodIndecies]
        if not AllElect:
            plotEEGs([cleanData[ix] for ix, electIX in enumerate(goodIndecies)],
                     tLabels[startIX:endIX],
                     [electLabels[electIX] for electIX in goodIndecies])

        for ix, electIX in enumerate(goodIndecies):
            sequ = np.array(cleanData[ix], dtype=np.float64)
            blinksBig, blinksDisBig, blinkIXsBig = findBlinks(blinkOutcomes[electIX]['original']['blinkWave'],
                                                    sequ, blinkDuration,
                                         sampleHz=sampleRate,
                                         tLabels=tLabels[startIX:endIX],
                                         verbose=7 if not AllElect else 0, electrode=electLabels[electIX])
            blinkOutcomes[electIX]['Big']['blinkWave'] = copy.deepcopy(blinkOutcomes[electIX]['original']['blinkWave'])
            blinkOutcomes[electIX]['Big']['blinks'] = blinksBig
            blinkOutcomes[electIX]['Big']['blinksIndecies'] = blinkIXsBig
            blinkOutcomes[electIX]['Big']['dissimilarity'] = blinksDisBig
            blinkOutcomes[electIX]['Big']['duration'] = blinkDurationMS
            print(f"{len(blinksBig)} Blinks per minute: {len(blinksBig)/((endTime-startTime)/60)}")
        if len(goodIndecies) == 1:
            electIX = goodIndecies[0]
            plotMotifMatches(cleanData[0], blinkOutcomes[electIX]['Big']['blinksIndecies'],
                             waveDuration,
                             title=f'Motif Match (Electrode {electLabels[electIX]})')
        else:
            if not AllElect:
                eeg = []
                for ix, electIX in enumerate(goodIndecies):
                    eeg.append(cleanData[ix])
                patches = [blinkOutcomes[electIX]['Big']['blinksIndecies'] for electIX in goodIndecies]
                AllBlinks = [electLabels[electIX] for electIX in goodIndecies]
                plotMotifMatchesMultiElectrodes(eeg, tLabels[startIX:endIX], patches, waveDuration,
                                                title="All electrodes All waves ''", electrodes=AllBlinks)
            for electIX in goodIndecies:
                print(f"{electLabels[electIX]}: {blinkOutcomes[electIX]['Big']['blinks']}")
            waveRespMetrics = plotSynchedMeanWaves([allData[electIX][startIX:endIX] for electIX in goodIndecies],
                                 tLabels[startIX:endIX],
                                 [blinkOutcomes[electIX]['Big']['blinksIndecies'] for electIX in goodIndecies],
                                 waveDuration,
                                 electrodes=[electLabels[electIX] for electIX in goodIndecies])


    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib import colors
    import mne
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    montage.ch_names[-1] = 'E129'
    testRaw.set_montage(montage, match_case=False)
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
    uu = mne.viz.plot_sensors(testRaw.info, ch_type="eeg",
                              axes=ax, ch_groups=chCategories,
                              cmap=cmapf, linewidth=0, pointsize=50,
                              show_names=False,
                              kind="topomap",
                              to_sphere=True, show=False)
    cbar = uu.colorbar(sm, ax=ax)
    uu.show()

    print("done")
    return


if __name__ == "__main__":
    params = sys.argv
    HARDCODE = True
    if HARDCODE:
        paramsDict = \
            {'test': [
                '--interactive', 'NO',
                '--dataFile', 'data/raw/ACL_035_raw.set',
                '--eventDuration', '300',
                '--sampleRate', '1000',
                '--channels', '14 8 1 All',
                '--badChannels', '44',
                '--learnStart', '707',
                '--learnStop', '721',
                '--findStart', '900',
                '--findStop', '1000'
            ],
            }
    params = paramsDict['test']
    main(params)