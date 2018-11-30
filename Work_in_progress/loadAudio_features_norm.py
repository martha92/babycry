import wave
import numpy as np
import utils
import librosa
import librosa.display
import numpy
from IPython import embed
import os
import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing


def load_audio(filename, mono=False, fs=44100):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()
        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None


def extract_mbe(_y, _sr, _nfft, _nb_mel):
    # spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length= 1024, power=2)
    # mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    # return np.log(np.dot(mel_basis, spec))
    spec = librosa.feature.melspectrogram(_y, sr=_sr, n_fft=nfft, hop_length=1024,
                                          n_mels=_nb_mel, fmax=22050, power=int(1))
    spec = librosa.power_to_db(spec)
    print(spec)

    return spec


# ###################################################################
#              Main script starts here
# ###################################################################

is_mono = False
__class_labels = {
    'hu': 0,
    'bu': 1,
    'bp': 2,
    'dc': 3,
    'ti': 4,
    'lo': 5,
    'ch': 6,
    'sc': 7,
    'dk': 8

}

# location of data.
folds_list = [1, 2, 3, 4]
evaluation_setup_folder = '/Users/marthagarcia/Documents/SFU_Master_BigData/Semester_1/Machine_Learning/Final_project/babycry/evaluation_folder'
audio_folder = '/Users/marthagarcia/Documents/SFU_Master_BigData/Semester_1/Machine_Learning/Final_project/babycry/Data/New_WAV'

# Output
feat_folder = '/Users/marthagarcia/Documents/SFU_Master_BigData/Semester_1/Machine_Learning/Final_project/babycry/features'
utils.create_folder(feat_folder)

# User set parameters
nfft = 2048
win_len = nfft
hop_len = win_len / 2
nb_mel_bands = 40
sr = 44100

# -----------------------------------------------------------------------
# Feature extraction and label generation
# -----------------------------------------------------------------------
# Load labels
desc_dict = {}
for audio_filename in os.listdir(audio_folder):
    name_file = audio_filename.split('.wav')
    name = name_file[0]
    print(name)
    label_info = name.split('-')
    size_tmp = len(label_info)
    _label = label_info[size_tmp - 1]
    age = label_info[size_tmp - 2]

    if name not in desc_dict:
        desc_dict[name] = [1, int(age), __class_labels[_label]]

# train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(1))
# evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))
# desc_dict = load_desc_file(train_file)
# desc_dict.update(load_desc_file(evaluate_file)) # contains labels for all the audio in the dataset


# Extract features for all audio files, and save it along with labels
for audio_filename in os.listdir(audio_folder):
    audio_file = os.path.join(audio_folder, audio_filename)
    print('Extracting features and label for : {}'.format(audio_file))
    y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
    mbe = None

    if is_mono:
        mbe = extract_mbe(y, sr, nfft, nb_mel_bands).T
    else:
        for ch in range(y.shape[0]):
            mbe_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands).T
            if mbe is None:
                mbe = mbe_ch
            else:
                mbe = np.concatenate((mbe, mbe_ch), 1)

    label = np.zeros((mbe.shape[0], len(__class_labels)))
    name_file = audio_filename.split('.wav')
    name = name_file[0]
    tmp_data = np.array(desc_dict[name])

    frame_start = np.floor(1 * sr / hop_len).astype(int)
    frame_end = np.ceil(26 * sr / hop_len).astype(int)
    # print(tmp_data[2:3])
    se_class = tmp_data[2:3].astype(int)
    # for ind, val in enumerate(se_class):
    #    label[frame_start[ind]:frame_end[ind], val] = 1
    tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
    np.savez(tmp_feat_file, mbe, label)
    # np.savez(mbe, label)

# -----------------------------------------------------------------------
# Feature Normalization
# -----------------------------------------------------------------------

for fold in folds_list:
    # train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(1))
    # evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))
    train_dict = desc_dict
    test_dict = desc_dict

    X_train, Y_train, X_test, Y_test = None, None, None, None
    for key in train_dict.keys():
        tmp_feat_file = os.path.join(feat_folder, (key + ".wav_bin.npz"))

        dmp = np.load(tmp_feat_file)
        tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
        if X_train is None:
            X_train, Y_train = tmp_mbe, tmp_label
        else:
            X_train, Y_train = np.concatenate((X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)

    for key in test_dict.keys():
        tmp_feat_file = os.path.join(feat_folder, (key + ".wav_bin.npz"))
        dmp = np.load(tmp_feat_file)
        tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
        if X_test is None:
            X_test, Y_test = tmp_mbe, tmp_label
        else:
            X_test, Y_test = np.concatenate((X_test, tmp_mbe), 0), np.concatenate((Y_test, tmp_label), 0)

    # Normalize the training data, and scale the testing data using the training data weights
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
    np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test)
    print('normalized_feat_file : {}'.format(normalized_feat_file))
