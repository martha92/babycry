import wave
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy
import soundfile as sf


class AudioAugmentation:
    def read_audio_file(self, file_path):
        data, smaple_rate = sf.read(file_path)
        return data, smaple_rate
    
    def write_audio_file(self, file, data, sample_rate):
        sf.write(file, data, sample_rate, subtype='PCM_16')
        
    def add_noise(self, data):
        noise = np.random.randn(data.shape[0], data.shape[1])
        data_noise = data + 0.005 * noise
        return data_noise
    
    def shift(self, data):
        y_shift = data.copy()
        timeshift_fac = 0.5 * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
        print("timeshift_fac = ", timeshift_fac)
        start = int(y_shift.shape[0] * timeshift_fac)
        print(start)
        if (start > 0):
            y = np.array([np.pad(y_shift[:, 0], (start, 0), mode='constant')[0:y_shift[:, 0].shape[0]], \
                          np.pad(y_shift[:, 1], (0, start), mode='constant')[0:y_shift[:, 1].shape[0]]])
        else:
            y = np.array([np.pad(y_shift[:, 0], (0, -start), mode='constant')[0:y_shift[:, 0].shape[0]], \
                          np.pad(data[:, 1], (0, -start), mode='constant')[0:data[:, 1].shape[0]]])
        return y.T
    
    def stretch(self, y, rate=1):
        data = np.array([librosa.effects.time_stretch(y[:, 0], rate), librosa.effects.time_stretch(y[:, 1], rate)]).T
        # data = librosa.effects.time_stretch(data, rate)
        return data


# Create a new instance from AudioAugmentation class
aa = AudioAugmentation()

# Read and produce augmentation files
list1 = os.listdir("ti")
print(list1)

# files = list(glob.glob(os.path.join("input",'*.*')))
# print(files)


for file in list1:
    if not file.startswith('.'):
        # print(file)
        data, sr = aa.read_audio_file("ti/" + file)
        # aa.plot_time_series(data)
        # Adding noise to sound
        data_noise = aa.add_noise(data)
        # aa.plot_time_series(data_noise)
        # Shifting the sound
        data_roll = aa.shift(data)
        # aa.plot_time_series(data_roll)
        # Stretching the sound
        data_stretch = aa.stretch(data, 0.8)
        # aa.plot_time_series(data_stretch)
        # Write generated cat sounds
        aa.write_audio_file('output/generated1_' + file, data_noise, sr)
        aa.write_audio_file('output/generated2_' + file, data_roll, sr)
        aa.write_audio_file('output/generated3_' + file, data_stretch, sr)

