from __future__ import print_function
import os
import numpy as np
import time
import sys
import keras
import matplotlib.pyplot as plot
import utils
import keras.backend as K
from sklearn.metrics import f1_score, accuracy_score

K.set_image_data_format('channels_first')
plot.switch_backend('agg')
sys.setrecursionlimit(10000)

home = '/home/ipsita_proff'

__class_labels = {
    0: 'hu',
    1: 'bu',
    2: 'bp',
    3: 'dc',
    4: 'ti',
    5: 'lo',
    6: 'ch',
    7: 'sc',
    8: 'dk'

}

__class_labels_desc = {
    'hu': 'hungry',
    'bu': 'needs burping',
    'bp': 'belly pain',
    'dc': 'discomfort',
    'ti': 'tired',
    'lo': 'lonely',
    'ch': 'cold/hot',
    'sc': 'scared',
    'dk': 'dont know'

}


def most_common(lst):
    return max(set(list(lst)), key=list(lst).count)


def load_data(_feat_folder, _mono, _fold=None):
    feat_file = home + '/babycry/features/mbe_bin_fold1.npz'
    dmp = np.load(feat_file)
    _X_train, _Y_train, _X_test, _Y_test, test_labels, f_train, f_test, seq_len = dmp['arr_0'], dmp['arr_1'], dmp[
        'arr_2'], dmp['arr_3'], dmp['arr_4'], dmp['arr_5'], dmp['arr_6'], dmp['arr_7']
    return _X_train, _Y_train, _X_test, _Y_test, test_labels, f_train, f_test, seq_len


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test


def getLabels(pred_labels, test_labels):
    index = 0
    filename = ""
    actual_label = ""
    for arr in pred_labels:
        _lblwithhigherprob = np.bincount(arr).argmax()
        print(_lblwithhigherprob)
        get_label = __class_labels[_lblwithhigherprob]
        get_label_desc = __class_labels_desc[get_label]
        # print(test_labels[''])
        for key, value in dict(np.ndenumerate(test_labels)).items():
            # value = dict()
            _index = str(index)
            filename = value[_index][0]
            actual_label = value[_index][1]
            index += 1
        # info_test = test_labels.get(index)
        print("File Name-> " + filename + " Predicted label-> " + get_label_desc + " Actual label-> " +
              __class_labels_desc[actual_label])


#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################

is_mono = False  # True: mono-channel input, False: binaural input

feat_folder = ''
nb_ch = 1 if is_mono else 2

fold = 1
print('\n\n----------------------------------------------')
print('FOLD: {}'.format(fold))
print('----------------------------------------------\n')
# Load feature and labels, pre-process it
X, Y, X_test, Y_test, test_labels, f_train, f_test, seq_len = load_data(feat_folder, is_mono, fold)
print(X_test.shape)

X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)
print(X_test.shape)
# Load model

model = keras.models.load_model('babycry_model_wip3_model.h5')

train_pred = model.predict(X)
pred = model.predict(X_test)
# Calculate the predictions on test data, in order to calculate ER and F scores
y_train = [most_common(Y.argmax(axis=-1)[i]) for i in range((Y.argmax(axis=-1)).shape[0])]
y_hat_train = [most_common(train_pred.argmax(axis=-1)[i]) for i in range((train_pred.argmax(axis=-1)).shape[0])]
y_test = [most_common(Y_test.argmax(axis=-1)[i]) for i in range((Y_test.argmax(axis=-1)).shape[0])]
y_hat_test = [most_common(pred.argmax(axis=-1)[i]) for i in range((pred.argmax(axis=-1)).shape[0])]
y_hat_prob_test = [np.mean(pred[i], axis=-2) for i in range((pred.argmax(axis=-1)).shape[0])]
print("Training Accuracy = {}, F1 score = {}".format(accuracy_score(y_train, y_hat_train),
                                                     f1_score(y_train, y_hat_train, average='weighted')))
print("Test Accuracy = {}, F1 score = {}".format(accuracy_score(y_test, y_hat_test),
                                                 f1_score(y_test, y_hat_test, average='weighted')))

for i in range(len(y_test)):
    print("\n\nFile: {}, Actual: {}, Predicted: {}\nPredicted Prob:\n".format(f_test[i], __class_labels_desc[__class_labels[y_test[i]]], __class_labels_desc[__class_labels[y_hat_test[i]]]))
    for j in range(len(__class_labels.keys())):
        print("{}: {}".format(__class_labels_desc[__class_labels[j]], y_hat_prob_test[i][j]))



# print(y_train)
# print(y_hat_train)
# print(f_train)

