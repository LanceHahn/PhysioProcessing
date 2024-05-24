import copy
import sys
import argparse
import json
import numpy as np
from mne.io.eeglab import read_raw_eeglab, read_epochs_eeglab
from blinkDection import (findBlinkWave, findBlinks, combineWaves, plotWaves, zeroOutOfRange,
                          plotMotifMatchesMultiElectrodes, plotEEGs, plotMotifMatches,
                          plotSynchedMeanWaves, stratifyForColors,
                          plotSensorStrengths)


def getChannels(askUser, electLabels, channelString, badChannelString):
    """
        Acquire channel list from user or passed parameters
    :param askUser: whether to ask the user for times
    :param electLabels: list of string electrode labels (e.g., 'E1')
    :param channelString: space-delimited list of numerical electrode labels
    that should be included in pipeline.  It may include 'ALL'
    :param badChannelString: space-delimited list of numerical electrode labels
     that should be excluded in pipeline.
    :return: AllElect - whether 'ALL' is included, goodChannels list of channels ['E1', 'E2', ...]
    """
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
    return AllElect, goodChannels


def getTimes(askUser, learnStartTime, learnStopTime):
    """
    Acquire start/stop times from user or passed parameters
    :param askUser: whether to ask the user for times
    :param learnStartTime: passed parameter time (s)
    :param learnStopTime: passed parameter time (s)
    :return: startTime, endTime
    """
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
    return startTime, endTime


def parse_args(params) -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--interactive', type=str, default='YES')
    parser.add_argument('--dynamicWindow', type=str, default='NO')
    parser.add_argument('--dataFile', type=str, default='data/raw/ACL_035_raw.set')
    parser.add_argument('--readTemplate', type=str, default='UNDEFINED.json')
    parser.add_argument('--writeTemplate', type=str, default='UNDEFINED.json')
    parser.add_argument('--sampleRate', type=int, default=1000)
    parser.add_argument('--eventDuration', type=int, default=300)
    parser.add_argument('--channels', type=str, default='14 8 1 All')
    parser.add_argument('--badChannels', type=str, default='44')
    parser.add_argument('--learnStart', type=int, default=707)
    parser.add_argument('--learnStop', type=int, default=721)
    parser.add_argument('--findStart', type=int, default=900)
    parser.add_argument('--findStop', type=int, default=1000)
    parser.add_argument('--pipeline', choices=['LEARN', 'FIND', 'ALL'], default='all')

    args = parser.parse_args(params)
    return args


def writeTemplateFile(dataIn, fName):
    """
    convert the numpy formatted data to something acceptable for JSON
    serialization and write the data to a JSON file
    :param dataIn: incoming data
    :param fName: the name of the file to write
    :return: None
    """
    dataOut = dict()
    for chan in dataIn.keys():
        dataOut[chan] = dict()
        for vers in dataIn[chan].keys():
            dataOut[chan][vers] = dict()
            for elem in dataIn[chan][vers].keys():
                if isinstance(dataIn[chan][vers][elem], np.ndarray):
                    dataOut[chan][vers][elem] = dataIn[chan][vers][elem].tolist()
                elif isinstance(dataIn[chan][vers][elem], list):
                    dataOut[chan][vers][elem] = dataIn[chan][vers][elem].copy()
                else:
                    dataOut[chan][vers][elem] = dataIn[chan][vers][elem]

    with open(fName, "w") as json_file:
        json.dump(dataOut, json_file)
    return


def readTemplateFile(fName):
    """
    read the JSON file and convert the 'blinkWave' data into numpy array
    :param fName: the name of the file to read
    :return: the data read from the file
    """
    with open(fName, "r") as json_file:
        dataIn = json.load(json_file)
    dataOut = dict()
    for chan in dataIn.keys():
        chan_I = int(chan)
        dataOut[chan_I] = dict()
        for vers in dataIn[chan].keys():
            dataOut[chan_I][vers] = dict()
            for elem in dataIn[chan][vers].keys():
                if elem == 'blinkWave':
                    dataOut[chan_I][vers][elem] = np.array(dataIn[chan][vers][elem])
                else:
                    dataOut[chan_I][vers][elem] = dataIn[chan][vers][elem]
    return dataOut


def extendWindow(expectedBlinks, delta, signals, signalDuration, sampleRate,
                 data, timeLabels, srcLabel, verbose):
    # EXTEND window until it alters the number of blinks discovered
    #expectedBlinks = len(blinks1)
    blinkCount = expectedBlinks
    signalsExt = dict()
    signalsExt['blinks'] = copy.deepcopy(signals['original']['blinks'])
    signalsExt['blinksIndecies'] = copy.deepcopy(signals['original']['blinksIndecies'])
    signalsExt['dissimilarity'] = copy.deepcopy(signals['original']['dissimilarity'])
    signalsExt['blinkWave'] = signals['original']['blinkWave']
    signalsExt['duration'] = signals['original']['duration']
    # Expanding the duration of the target signal only makes sense if there
    # is more than one signal already detected because expanding the
    # duration of the target decreases the likelihood of multiple signals
    # matching and 1 is the lowest possible count.
    if blinkCount > 1:
        waveDuration = signalDuration
        while blinkCount == expectedBlinks:
            waveDuration += delta
            blinkDuration = waveDuration / sampleRate  # unit: seconds
            print(f"Considering wave duration of {waveDuration}...")
            sequ = data

            # Find initial blink wave as best duplicated sequence.
            blinkWaveW0 = findBlinkWave(sequ, blinkDuration,
                                        sampleHz=sampleRate,
                                        tLabels=timeLabels,
                                        verbose=0, electrode=srcLabel)

            blinksW0, blinkDisW0, _ = findBlinks(blinkWaveW0, sequ, blinkDuration,
                                                 sampleHz=sampleRate,
                                                 tLabels=timeLabels,
                                                 verbose=0,
                                                 electrode=srcLabel)
            startTime, endTime = timeLabels[0], timeLabels[-1]
            startIndecies = [np.where(timeLabels == b)[0][0] for b in blinksW0]
            newBlinkWave = combineWaves([sequ[start: start + waveDuration]
                                         for start in startIndecies])
            if blinkCount == expectedBlinks:
                print(f"{len(blinksW0)} Blinks per minute: {len(blinksW0) / ((endTime - startTime) / 60)}")
                # Try again with new updated wave
                print(f"Repeat blink discovery with wave generated from "
                      f"{len(blinksW0)} detected waves.")
                blinksW2, blinkDisW2, blinksIXsW2 = findBlinks(newBlinkWave, sequ,
                                                               blinkDuration,
                                                               sampleHz=sampleRate,
                                                               tLabels=timeLabels,
                                                               verbose=verbose,
                                                               electrode=srcLabel)
                blinkCount = len(blinksW2)
                if blinkCount == expectedBlinks:
                    signalsExt['blinks'] = copy.deepcopy(blinksW2)
                    signalsExt['blinksIndecies'] = copy.deepcopy(blinksIXsW2)
                    signalsExt['dissimilarity'] = copy.deepcopy(blinkDisW2)
                    signalsExt['blinkWave'] = copy.deepcopy(newBlinkWave)
                    signalsExt['duration'] = waveDuration
                else:
                    print(f"{srcLabel} Stop Extension: Signal count changed from {expectedBlinks} to {blinkCount}.")
            else:
                print(f"{srcLabel} Stop Extension: Signal count changed from {expectedBlinks} to {blinkCount}.")
        waveDuration -= delta
    return signalsExt


def FindEvents(signals, askUser, findStartTime, findStopTime,
               tLabels, sampleRate,
               data, AllElect,
               electLabels, goodIndecies, blinkDurationMS):
    ### apply wave detection to full range of data
    print("Going Big (longer timeline)")
    print(f"Data time range is from 0 to {int(len(tLabels)/sampleRate)} seconds")
    blinkDuration = blinkDurationMS / sampleRate
    startTime, endTime = getTimes(askUser, findStartTime, findStopTime)
    startIX = startTime * sampleRate
    endIX = endTime * sampleRate
    cleanData = [zeroOutOfRange(data[electIX][startIX:endIX]) for electIX in goodIndecies]
    if not AllElect:
        print("Generating plot of electrode signal(s)")
        plotEEGs([cleanData[ix] for ix, electIX in enumerate(goodIndecies)],
                 tLabels[startIX:endIX],
                 [electLabels[electIX] for electIX in goodIndecies])

    for ix, electIX in enumerate(goodIndecies):
        sequ = np.array(cleanData[ix], dtype=np.float64)
        blinksBig, blinksDisBig, blinkIXsBig = (
            findBlinks(signals[electIX]['original']['blinkWave'],
                       sequ, blinkDuration, sampleHz=sampleRate,
                       tLabels=tLabels[startIX:endIX],  verbose=7 if not AllElect else 0, electrode=electLabels[electIX]))
        signals[electIX]['Big']['blinkWave'] = copy.deepcopy(signals[electIX]['original']['blinkWave'])
        signals[electIX]['Big']['blinks'] = blinksBig
        signals[electIX]['Big']['blinksIndecies'] = blinkIXsBig
        signals[electIX]['Big']['dissimilarity'] = blinksDisBig
        signals[electIX]['Big']['duration'] = blinkDurationMS
        print(f"{len(blinksBig)} Blinks per minute: {len(blinksBig)/((endTime-startTime)/60)}")
    if len(goodIndecies) == 1:
        electIX = goodIndecies[0]
        plotMotifMatches(cleanData[0], signals[electIX]['Big']['blinksIndecies'],
                         blinkDurationMS,
                         title=f'Motif Match (Electrode {electLabels[electIX]})')
    else:
        if not AllElect:
            eeg = []
            for ix, electIX in enumerate(goodIndecies):
                eeg.append(cleanData[ix])
            patches = [signals[electIX]['Big']['blinksIndecies'] for electIX in goodIndecies]
            AllBlinks = [electLabels[electIX] for electIX in goodIndecies]
            plotMotifMatchesMultiElectrodes(eeg, tLabels[startIX:endIX], patches, blinkDurationMS,
                                            title="All electrodes All waves ''", electrodes=AllBlinks)
        for electIX in goodIndecies:
            print(f"{electLabels[electIX]}: {signals[electIX]['Big']['blinks']}")
        waveRespMetrics = plotSynchedMeanWaves([data[electIX][startIX:endIX] for electIX in goodIndecies],
                             tLabels[startIX:endIX],
                             [signals[electIX]['Big']['blinksIndecies'] for electIX in goodIndecies],
                             blinkDurationMS,
                             electrodes=[electLabels[electIX] for electIX in goodIndecies])

    return signals, waveRespMetrics

def main(params):

    # incorporate user's parameters
    args = parse_args(params)
    fnameSetRaw = args.dataFile
    askUser = args.interactive.upper() == 'YES'
    dynamicWindow = args.dynamicWindow.upper() == 'YES'
    sampleRate = args.sampleRate
    blinkDurationMS = args.eventDuration
    channelString = args.channels
    badChannelString = args.badChannels
    learnStartTime = args.learnStart
    learnStopTime = args.learnStop
    findStartTime = args.findStart
    findStopTime = args.findStop
    pipeline = args.pipeline
    doFindEvents = pipeline in {'FIND', 'ALL'}
    learn = pipeline in {'LEARN', 'ALL'}
    readTemplate = args.readTemplate
    writeTemplate = args.writeTemplate

    # Read data file and gather data values, timeframe and electrode labels
    if askUser:
        fname = input(f"Name of the data file (default: {fnameSetRaw}? ")
        if len(fname) > 1:
            fnameSetRaw = fname
    if askUser:
        fname = input(f"Name of the data file (default: {readTemplate}? ")
        if len(fname) > 1:
            readTemplate = fname
    if not learn and not readTemplate:
        print(f"No template signal wave defined.")
        print("Either a signal wave template file is needed when skipping the learning phase. ")

    testRaw = read_raw_eeglab(input_fname=fnameSetRaw)
    allData = testRaw.get_data()
    tLabels = testRaw.times
    electLabels = testRaw.ch_names
    print(f"{len(electLabels)} Electrode labels found: {electLabels}")
    print(f"{len(tLabels)} Time labels found: {tLabels}")

    AllElect, goodChannels = getChannels(askUser, electLabels,
                                         channelString, badChannelString)
    goodIndecies = [electLabels.index(x) for x in goodChannels]

    waveDuration = blinkDurationMS
    print(f"Sample Rate: {sampleRate}    temporal window (ms): {waveDuration}    ")
    startTime, endTime = getTimes(askUser, learnStartTime, learnStopTime)
    startIX = startTime * sampleRate
    endIX = endTime * sampleRate

    if not AllElect:
        print("Generating plot of electrode signal(s)")
        plotEEGs([allData[electIX][startIX:endIX] for electIX in goodIndecies],
                 tLabels[startIX:endIX],
                 [electLabels[electIX] for electIX in goodIndecies])

    blinkDuration = blinkDurationMS / sampleRate  # event duration in sample count
    if learn:
        blinkOutcomes = {}
        for electIX in goodIndecies:
            print(f"\n*** Processing electrode {electLabels[electIX]}")
            blinkOutcomes[electIX] = {
                'original': {},
                'extended': {},
                'Big': {}
            }

            # grab limited temporal window
            sequ = allData[electIX][startTime * sampleRate:endTime * sampleRate]
            # Find initial signal event wave as best duplicated sequence.
            blinkWave = findBlinkWave(sequ, blinkDuration,
                                      sampleHz=sampleRate,
                                      tLabels=tLabels[startTime*1000:endTime*1000],
                                      verbose=2 if not AllElect else 0, electrode=electLabels[electIX])

            # Find all instances of this signal event within the time range
            blinks, blinkDis, _ = (
                findBlinks(blinkWave, sequ, blinkDuration,
                           sampleHz=sampleRate,
                           tLabels=tLabels[startTime*1000:endTime*1000],
                           verbose=3 if not AllElect else 0,
                           electrode=electLabels[electIX]))
            startIndecies = [np.where(tLabels == b)[0][0] - (startTime*1000) for b in blinks]
            newBlinkWave = combineWaves([sequ[start: start + blinkDurationMS]
                                         for start in startIndecies])
            print(f"Events per minute: {len(blinks)/((endTime-startTime)/60)}")

            # Try again with new updated wave
            print(f"Repeat event discovery with wave generated from {len(blinks)} "
                  f"detected waves.")

            blinks1, blinksDis1, blinkIXs1 = (
                findBlinks(newBlinkWave, sequ, blinkDuration, sampleHz=sampleRate,
                           tLabels=tLabels[startTime*1000:endTime*1000],
                           verbose=4 if not AllElect else 0,
                           electrode=electLabels[electIX]))
            blinkOutcomes[electIX]['original']['blinkWave'] = copy.deepcopy(newBlinkWave)
            blinkOutcomes[electIX]['original']['blinks'] = blinks1
            blinkOutcomes[electIX]['original']['blinksIndecies'] = blinkIXs1
            blinkOutcomes[electIX]['original']['dissimilarity'] = blinksDis1
            blinkOutcomes[electIX]['original']['duration'] = blinkDurationMS
            print(f"# {electLabels[electIX]} Blinks per minute ({len(blinks1)} "
                  f"blinks): {len(blinks1)/((endTime-startTime)/60)}")

            if dynamicWindow:
                # EXTEND window until it alters the number of blinks discovered
                delta = 30
                print(f"The initial time window is being extended from a time "
                      f"window of {blinkDurationMS} by steps of "
                      f"{delta} ms until it alters the number of blinks...")
                blinkOutcomes[electIX]['extended'] = (
                    extendWindow(len(blinks1), delta, blinkOutcomes[electIX],
                                 blinkDurationMS, sampleRate,
                                 allData[electIX][startTime * sampleRate:endTime * sampleRate],
                                 tLabels[startTime * 1000:endTime * 1000],
                                 electLabels[electIX],
                                 verbose=3 if not AllElect else 0
                                 ))
                if not AllElect:
                    plotWaves([blinkOutcomes[electIX]['extended']['blinkWave']],
                              xLabels=[],
                              labels=[f'blink on {goodChannels[goodIndecies.index(electIX)]}'],
                              title=f"Extended wave {waveDuration} with {len(blinkOutcomes[electIX]['extended']['blinks'])} blinks"
                                    f" on {goodChannels[goodIndecies.index(electIX)]}")
            else:
                print(f"dynamicWindow is False which means that the wave will be"
                      f" held to the {blinkDurationMS} ms expected time window.")
                if not AllElect:
                    plotWaves([blinkOutcomes[electIX]['original']['blinkWave']], xLabels=[],
                              labels=[f'blink on {goodChannels[goodIndecies.index(electIX)]}'],
                              title=f"Final wave {waveDuration} with {len(blinkOutcomes[electIX]['original']['blinks'])} events"
                                    f" on {goodChannels[goodIndecies.index(electIX)]}")

        if dynamicWindow:
            if len(goodChannels) < 10:
                title = f"Final wave(s) {', '.join([str(blinkOutcomes[k]['extended']['duration']) for k in blinkOutcomes.keys()])}ms with "+\
                            f"{', '.join([str(len(blinkOutcomes[k]['extended']['blinks'])) for k in blinkOutcomes.keys()])} events" +\
                            f" on {', '.join(goodChannels)}"
            else:
                title = f"Final {len(goodChannels)} wave(s) {sum([blinkOutcomes[k]['extended']['duration'] for k in blinkOutcomes.keys()])/len(blinkOutcomes)} ms with "+\
                            f"{sum([len(blinkOutcomes[k]['extended']['blinks']) for k in blinkOutcomes.keys()])/len(blinkOutcomes)} events"
            plotWaves([blinkOutcomes[ix]['extended']['blinkWave'] for ix in goodIndecies], xLabels=[],
                      labels=goodChannels,
                      title=title)
        else:
            if len(goodChannels) < 10:
                title = f"Final wave(s) {blinkDurationMS}ms with " +\
                            f"{', '.join([str(len(blinkOutcomes[k]['original']['blinks'])) for k in blinkOutcomes.keys()])} blinks"+\
                            f" on {', '.join(goodChannels)}"
            else:
                title = f"Final {len(goodChannels)}wave(s) {blinkDurationMS} ms with " +\
                            f"{sum([len(blinkOutcomes[k]['original']['blinks']) for k in blinkOutcomes.keys()])/len(blinkOutcomes)} blinks"
            plotWaves([blinkOutcomes[ix]['original']['blinkWave'] for ix in goodIndecies],
                      xLabels=[], labels=goodChannels,
                      title=title)
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
    else:
        # Open the JSON file and load its contents
        blinkOutcomes = readTemplateFile(readTemplate)

    if doFindEvents:
        ### apply wave detection to full range of data
        blinkOutcomes, waveRespMetrics = FindEvents(blinkOutcomes, askUser, findStartTime, findStopTime,
                                   tLabels, sampleRate,
                                   allData, AllElect,
                                   electLabels, goodIndecies, blinkDurationMS)
    if len(writeTemplate) > 1:
        print(f"Writing wave templates to {writeTemplate}")
        writeTemplateFile(blinkOutcomes, writeTemplate)
    plotSensorStrengths(goodChannels, waveRespMetrics, electLabels,
                        testRaw.set_montage, testRaw.info)
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
                '--findStop', '1000',
                '--dynamicWindow', 'YES',
                '--pipeline', 'ALL',  # 'LEARN', 'FIND', 'ALL'
                '--readTemplate', 'waveTemplate.json',
                '--writeTemplate', 'waveTemplate.json',
            ],
            }
    params = paramsDict['test']
    main(params)