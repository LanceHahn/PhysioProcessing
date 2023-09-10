
from scipy.io import wavfile

samplerate, data = wavfile.read('BYB_Recording_2023-09-08_11_35_48.wav') # read data file to find num of samples per a second and the amplitude at each sample
length = data.shape[0] / samplerate; # calculate the length of the audio file in seconds
print(f"{length}s")
print(f"Sample Rate = {samplerate}Hz") # number of samples per a second
print(f"Number of Samples = {data.shape[0]}") # number of samples in data set
maxAmplitude = max(data) # find the amplitude at the highest point
minAmplitude = min(data) # find the amplitude at the lowest point
print(f"Highest Amplitude = {maxAmplitude}")
print(f"Lowest Amplitude = {minAmplitude}")
for i in range(data.shape[0]):
    if maxAmplitude == data[i]: # compare each sample's amplitude to test if it is the maximum
        print(f"max {i / samplerate}s") # display when at maximum point
    elif minAmplitude == data[i]: # compare each sample's amplitude to test if it is the minimum
        print(f"min {i / samplerate}s") # display when at minimum point

