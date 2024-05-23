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

# Convert NumPy arrays to Python lists
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

def convert_np_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
# Define a custom decoder function to handle conversion of lists to NumPy arrays
def custom_decoder(obj):
    if '__ndarray__' in obj:
        data = np.frombuffer(bytes.fromhex(obj['__ndarray__']['data']), dtype=obj['__ndarray__']['dtype'])
        return data.reshape(obj['__ndarray__']['shape'])
    return obj
def main(params):

    # incorporate user's parameters
    args = parse_args(params)
    fnameSetRaw = args.dataFile
    askUser = args.interactive.upper() == 'YES'
    extendWindow = args.dynamicWindow.upper() == 'YES'
    sampleRate = args.sampleRate
    blinkDurationMS = args.eventDuration
    channelString = args.channels
    badChannelString = args.badChannels
    learnStartTime = args.learnStart
    learnStopTime = args.learnStop
    findStartTime = args.findStart
    findStopTime = args.findStop
    pipeline = args.pipeline
    goBig = pipeline in {'FIND', 'ALL'}
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
            print(f"{len(blinks1)} Blinks per minute: {len(blinks1)/((endTime-startTime)/60)}")
            startIndecies = [np.where(tLabels == b)[0][0] - (startTime * 1000) for
                             b in blinks1]

            if extendWindow:
                # EXTEND window until it alters the number of blinks discovered
                delta = 30
                print(f"The initial time window is being extended from a time "
                      f"window of {blinkDurationMS} by steps of "
                      f"{delta} ms until it alters the number of blinks...")
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
                print(f"extendWindow is False which means that the wave will "
                      f"be held to the {blinkDurationMS} ms expected time window.")
                if not AllElect:
                    plotWaves([blinkOutcomes[electIX]['original']['blinkWave']], xLabels=[],
                              labels=[f'blink on {goodChannels[goodIndecies.index(electIX)]}'],
                              title=f"Final wave {waveDuration} with {len(blinkOutcomes[electIX]['original']['blinks'])} blinks"
                                    f" on {goodChannels[goodIndecies.index(electIX)]}")

        if extendWindow:
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
    else:
        # Open the JSON file and load its contents
        blinkOutcomes = readTemplateFile(readTemplate)

    if goBig:
        ### apply wave detection to full range of data
        print("Going Big (longer timeline)")
        print(f"Data time range is from 0 to {int(len(tLabels)/sampleRate)} seconds")

        startTime, endTime = getTimes(askUser, findStartTime, findStopTime)
        startIX = startTime * sampleRate
        endIX = endTime * sampleRate
        cleanData = [zeroOutOfRange(allData[electIX][startIX:endIX]) for electIX in goodIndecies]
        if not AllElect:
            print("Generating plot of electrode signal(s)")
            plotEEGs([cleanData[ix] for ix, electIX in enumerate(goodIndecies)],
                     tLabels[startIX:endIX],
                     [electLabels[electIX] for electIX in goodIndecies])

        for ix, electIX in enumerate(goodIndecies):
            sequ = np.array(cleanData[ix], dtype=np.float64)
            blinksBig, blinksDisBig, blinkIXsBig = (
                findBlinks(blinkOutcomes[electIX]['original']['blinkWave'],
                           sequ, blinkDuration, sampleHz=sampleRate,
                           tLabels=tLabels[startIX:endIX],  verbose=7 if not AllElect else 0, electrode=electLabels[electIX]))
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
                '--pipeline', 'ALL',  # 'LEARN', 'FIND', 'ALL'
                '--readTemplate', 'waveTemplate.json',
                '--writeTemplate', 'waveTemplate.json',
            ],
            }
    params = paramsDict['test']
    main(params)