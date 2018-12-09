from __future__ import print_function
from google.colab import files
import os
import numpy as np
import time
import sys
import json
import keras
import matplotlib.pyplot as plot
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, \
    Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras_self_attention import SeqSelfAttention
from keras.models import Model
from sklearn.metrics import confusion_matrix
import metrics
import utils
from IPython import embed
import keras.backend as K

K.set_image_data_format('channels_first')
plot.switch_backend('agg')
sys.setrecursionlimit(10000)
from google.colab import drive

drive.mount('/content/gdrive')

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


def load_data(_feat_folder, _mono, _fold=None):
    feat_file = 'mbe_bin_fold1.npz'
    dmp = np.load(feat_file)
    _X_train, _Y_train, _X_test, _Y_test, test_labels = dmp['arr_0'], dmp['arr_1'], dmp['arr_2'], dmp['arr_3'], dmp[
        'arr_4']
    return _X_train, _Y_train, _X_test, _Y_test, test_labels


def get_model(data_in, data_out, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):
    spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_x = spec_start
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=2)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((data_in.shape[-2], -1))(spec_x)

    for _r in _rnn_nb:
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul')(spec_x)
        SeqSelfAttention(
            attention_width=3,
            attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
            attention_activation=None,
            kernel_regularizer=keras.regularizers.l2(1e-6),
            use_attention_bias=False,
            name='Attention',
        )(spec_x)

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = Model(inputs=spec_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy')
    _model.summary()
    return _model


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label='f')
    plot.plot(range(_nb_epoch), _er, label='er')
    plot.legend()
    plot.grid(True)

    plot.savefig('/content/gdrive/My Drive/ML_FinalProject/models_Multi_AM/' + __fig_name + extension)
    plot.close()
    print('figure name : {}'.format(__fig_name))


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
__fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))

nb_ch = 1 if is_mono else 2
batch_size = 64  # Decrease this if you want to run on smaller GPU's
seq_len = 256  # Frame sequence length. Input to the CRNN.
nb_epoch = 500  # Training epochs
patience = int(0.25 * nb_epoch)  # Patience for early stopping

# Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
# Make sure the nfft and sr are the same as in feature.py
sr = 44100
nfft = 2048
frames_1_sec = int(sr / (nfft / 2.0))

print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

# Folder for saving model and training curves
# __models_dir = ''
# utils.create_folder(__models_dir)

# CRNN model definition
cnn_nb_filt = 128  # CNN filter size
cnn_pool_size = [5, 2, 2]  # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
rnn_nb = [32, 32]  # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
fc_nb = [32]  # Number of FC nodes.  Length of fc_nb =  number of FC layers
dropout_rate = 0.5  # Dropout after each layer
print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
    cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

avg_er = list()
avg_f1 = list()
for fold in [1]:
    print('\n\n----------------------------------------------')
    print('FOLD: {}'.format(fold))
    print('----------------------------------------------\n')
    # Load feature and labels, pre-process it
    X, Y, X_test, Y_test, test_labels = load_data(feat_folder, is_mono, fold)

    X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)

    # Load model
    model = get_model(X, Y, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb)

    # Training
    best_epoch, pat_cnt, best_er, f1_for_best_er, best_conf_mat = 0, 0, 99999, None, None
    tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch, [
        0] * nb_epoch
    posterior_thresh = 0.5
    path = '{}_fold_{}_model.h5'.format(__fig_name, '1')

    for i in range(nb_epoch):
        print('Epoch : {} '.format(i), end='')
        hist = model.fit(
            X, Y,
            batch_size=batch_size,
            validation_data=[X_test, Y_test],
            epochs=1,
            verbose=2
        )
        val_loss[i] = hist.history.get('val_loss')[-1]
        tr_loss[i] = hist.history.get('loss')[-1]

        # Calculate the predictions on test data, in order to calculate ER and F scores
        pred = model.predict(X_test)
        # pred_labels = (pred.argmax(axis=-1)).printoptions(precision=10000000)
        # pred_labels = np.array2string((pred.argmax(axis=-1)), precision=10000000, separator=',')
        # pred_labels = np.asarray((pred.argmax(axis=-1)))
        pred_labels = (pred.argmax(axis=-1))
        # text=np.array2string(pred_labels)#.set_printoptions(threshold=np.inf)
        # f2 = open('/content/gdrive/My Drive/ML_FinalProject/models/predictedlabels.txt', 'w')

        # f2.write(str(pred_labels))
        # f2.close()
        # getLabels(pred_labels, test_labels)

        pred_thresh = pred > posterior_thresh
        score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)

        f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
        er_overall_1sec_list[i] = score_list['er_overall_1sec']
        pat_cnt = pat_cnt + 1

        # Calculate confusion matrix
        test_pred_cnt = np.sum(pred_thresh, 2)
        Y_test_cnt = np.sum(Y_test, 2)
        conf_mat = confusion_matrix(Y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
        conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))

        if er_overall_1sec_list[i] < best_er:
            best_conf_mat = conf_mat
            best_er = er_overall_1sec_list[i]
            f1_for_best_er = f1_overall_1sec_list[i]

            model.save('/content/gdrive/My Drive/ML_FinalProject/models_Multi_AM/' + path)

            best_epoch = i
            pat_cnt = 0

        print('tr Er : {}, val Er : {}, F1_overall : {}, ER_overall : {} Best ER : {}, best_epoch: {}'.format(
            tr_loss[i], val_loss[i], f1_overall_1sec_list[i], er_overall_1sec_list[i], best_er, best_epoch))
        plot_functions(nb_epoch, tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, '_fold_{}'.format(fold))
        if pat_cnt > patience:
            break
    avg_er.append(best_er)
    avg_f1.append(f1_for_best_er)
    print('saved model for the best_epoch: {} with best_f1: {} f1_for_best_er: {}'.format(
        best_epoch, best_er, f1_for_best_er))
    print('best_conf_mat: {}'.format(best_conf_mat))
    # print('best_conf_mat_diag: {}'.format(np.diag(best_conf_mat)))

print('\n\nMETRICS : avg_er: {}, avg_f1: {}'.format(avg_er, avg_f1))
print('MODEL AVERAGE : avg_er: {}, avg_f1: {}'.format(np.mean(avg_er), np.mean(avg_f1)))
