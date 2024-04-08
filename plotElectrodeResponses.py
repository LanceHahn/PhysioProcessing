import os
import pwd
from mne.io.eeglab import read_raw_eeglab, read_epochs_eeglab
import matplotlib.pyplot as plt
import numpy as np
from blinkDection import findBlinkWave, findBlinks, combineWaves, plotWaves
def get_username():
    return pwd.getpwuid(os.getuid())[0]

if get_username() == 'lance':
    basePath = '/Users/lance/Documents/GitHub/mne-python/'
    fnameSet = 'data/filtered and epoched not baseline corrected/ACL_035_filt_epoch_nobase.set'
    fnameSetFiltered = 'data/filtered/ACL_035_filt.set'
    fnameSetRaw = 'data/raw/ACL_035_raw.set'
else:
    basePath = '/Users/add32662/OneDrive - Western Kentucky University/Pycharm/mne-python-main/'
    fnameSet = 'mne/data/filtered and epoched not baseline corrected/ACL_035_filt_epoch_nobase.set'
    fnameSetFiltered = 'mne/data/filtered/ACL_035_filt.set'
    fnameSetRaw = 'mne/data/raw/ACL_035_raw.set'


#testEpochs = read_epochs_eeglab(input_fname=fnameSet)
testRaw = read_raw_eeglab(input_fname=fnameSetRaw)

allData = testRaw.get_data()
tLabels = testRaw.times
electLabels = testRaw.ch_names
print(f"{len(electLabels)} Electrode labels found: {electLabels}")
print(f"{len(tLabels)} Time labels found: {tLabels}")
offset = 0 # 0.1
base = 0.0
maxReal = 0.01
sampleRate = 1000
# goodChannels = [x for x in electLabels if x[-1] == '1']
# goodChannels = ["E14", "E21"]
channelString = input("Provide a space-delimited list of channel numbers (defaulty 14): ")
if len(channelString) < 1:
    channelString = '14'
goodChannels = channelString.split(' ')
goodChannels = ["E" + ch for ch in goodChannels]
startTime = int(60 * 21)
endTime = int(60 * 22)
sampleRate = 1000
timeString = input("Provide a start time in seconds (default: 707):")
if len(timeString) < 1:
    timeString = '707'
startTime = int(timeString)
timeString = input("Provide a end time in seconds (default: 721):")
if len(timeString) < 1:
    timeString = '721'
endTime = int(timeString)
print("Generating plots..")

for electIX in range(len(electLabels)):
    if not goodChannels or electLabels[electIX] in goodChannels:
        print(f"{electLabels[electIX]} ",end="")
        adjusted = [x if x < maxReal else None for x in allData[electIX]]
        maxV = max(adjusted[startTime * 1000:endTime * 1000])
        minV = min(adjusted[startTime * 1000:endTime * 1000])
        adjusted = [x - maxV + base if x < maxReal else None for x in allData[electIX]]
        #adjusted = [x for x in allData[electIX]]
        #plt.plot(tLabels, adjusted, label=electLabels[electIX])
        plt.plot(tLabels[startTime*1000:endTime*1000],
                 adjusted[startTime*1000:endTime*1000],
                 label=electLabels[electIX])
        base += -0.0003 + minV -maxV # offset

plt.title('initial EEG plot')
plt.legend()
plt.show()

blinkDurationMS = 150
blinkDuration = blinkDurationMS / sampleRate  #150 ms #0.3 # 0.1  # unit: seconds
interBlink = 60 / 12  # 12 times per minute unit: seconds
dwnSample = 1
for electIX in [x for x in range(len(electLabels)) if electLabels[x] in goodChannels]:
    if dwnSample > 1:
        sequ = [allData[electIX][x] for x in range(startTime*sampleRate, endTime*sampleRate, dwnSample)]
    else:
        sequ = allData[electIX][startTime * sampleRate:endTime * sampleRate]
    # Find initial blink wave as best duplicated sequence.
    blinkWave = findBlinkWave(sequ, blinkDuration, interBlink,
                              sampleHz=int(sampleRate/dwnSample),
                              tLabels=tLabels[startTime*1000:endTime*1000],
                              verbose=7)

    blinks = findBlinks(blinkWave, sequ, blinkDuration,
                        sampleHz=int(sampleRate/dwnSample),
                        tLabels=tLabels[startTime*1000:endTime*1000],
                        verbose=7)
    startIndecies = [np.where(tLabels == b)[0][0] - (startTime*1000) for b in blinks]
    newBlinkWave = combineWaves([sequ[start: start + blinkDurationMS]
                                 for start in startIndecies])
    plotWaves([blinkWave, newBlinkWave], xLabels=[],
              labels=["Blink", "NewBlink"])
    print(f"Blinks per minute: {len(blinks)/((endTime-startTime)/60)}")

    # Try again with new updated wave
    print(f"Repeat blink discover with wave generated from {len(blinks)} "
          f"detected waves.")
    blinks2 = findBlinks(newBlinkWave, sequ, blinkDuration,
                        sampleHz=int(sampleRate/dwnSample),
                        tLabels=tLabels[startTime*1000:endTime*1000],
                        verbose=7)
    print(f"Blinks per minute: {len(blinks2)/((endTime-startTime)/60)}")
akey = input('press return when ready to quit')
print("done")