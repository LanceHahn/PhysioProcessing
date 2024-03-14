from mne.io.eeglab import read_raw_eeglab, read_epochs_eeglab
import matplotlib.pyplot as plt

basePath = '/Users/add32662/OneDrive - Western Kentucky University/Pycharm/mne-python-main/'
fnameSet = 'mne/data/filtered and epoched not baseline corrected/ACL_035_filt_epoch_nobase.set'
fnameSetFiltered = 'mne/data/filtered/ACL_035_filt.set'
fnameSetRaw = 'mne/data/raw/ACL_035_raw.set'

testEpochs = read_epochs_eeglab(input_fname=fnameSet)
testRaw = read_raw_eeglab(input_fname=fnameSetRaw)

allData = testRaw.get_data()
tLabels = testRaw.times
electLabels = testRaw.ch_names
print(f"{len(electLabels)} Electrode labels found: {electLabels}")
print(f"{len(tLabels)} Time labels found: {tLabels}")
offset = 0 # 0.1
base = 0.0
maxReal = 0.01
goodChannels = [x for x in electLabels if x[-1] == '1']
goodChannels = ["E1"]
channelString = input("Provide a list of channel numbers: ")
print(channelString)
goodChannels = channelString.split(' ')
goodChannels = ["E" + ch for ch in goodChannels]
print("Generating plots..")
print(goodChannels)
startTime = int(input("Provide a start time in seconds:"))
endTime = int(input("Provide a end time in seconds:"))

for electIX in range(len(electLabels)):
    if not goodChannels or electLabels[electIX] in goodChannels:
        print(f"{electLabels[electIX]} ",end="")
        adjusted = [x if x < maxReal else None for x in allData[electIX]]
        maxV = max(adjusted[startTime * 1000:endTime * 1000])
        minV = min(adjusted[startTime * 1000:endTime * 1000])
        adjusted = [x - maxV + base if x < maxReal else None for x in allData[electIX]]
        #adjusted = [x for x in allData[electIX]]
        #plt.plot(tLabels, adjusted, label=electLabels[electIX])
        plt.plot(tLabels[startTime*1000:endTime*1000], adjusted[startTime*1000:endTime*1000], label=electLabels[electIX])
        base += -0.0003 + minV -maxV # offset

plt.title('initial EEG plot')
plt.legend()
plt.show()
akey = input('press return when ready to quit')
print("done")