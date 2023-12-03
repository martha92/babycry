## BabyCry

Baby cry is the primary way that an infant communicates with the caregivers. 
There are several researches that convey the idea that babies produce different sounds to convey their emotions.
However, the task of recognizing these different cries and response accordingly is overwhelming for first-time parents. 
Thus, remote monitor is one common application of automatic infant cry recognition system. 

> Classify the baby cry sound into 8 different categories (hungry, needs burping, scared, belly pain, discomfort, cold/hot, lonely, tired).

To address this we explore a number of deep learning approaches such as Convolutional neural networks (CNN), 
Recurrent Neural Networks (RNN) and Attention Model to achieve best result.

## Data Source

**Dataset** The primary data source for our project is <a href="https://github.com/gveres/donateacry-corpus#donateacry-corpus">donateacry-corpus</a>.
The audio files contain baby cry samples of infants from 0 to 2 years old, with the corresponding tagging information (the suspected reasons of cry) encoded in the filenames. 
Each tag is an abbreviation which uniquely identifies one of the 9 reasons (see Table 1) for the cry audio. <br><br>
The dataset consists of 716 unmodified audio files in Core Audio Format (caf extension) and 3gp format. 
Converted all files to WAV file format so that it could be easily read and interconverted by Python audio libraries (librosa, Wave and SoundFile).

**Data Cleaning** The following steps have been followed to extract useful data from the source repository
* Filter out audio files where the cry was dominant as in many samples the cry was not audible.
* A more detailed analysis of the dataset shows that the data is not uniformly distributed in all classes.
Out of the 700 samples we had, more than 400 samples were tagged for hunger,
the rest 300 comprised of the other remaining classes in varying numbers.
Thus, it was a **Class Imbalance** problem which was dealt with by **data augmentation** and changing the **performance metrics**.

|Tags|Classes/Reason for Cry|Original Audio Count|Counts after cleaning|Audio used after including augmented data|
|:----|:----|:----|:----|:----|
|hu|Hungry|485|350|50|
|bu|Needs to burp|8|4|21|
|bp|Belly pain|10|7|24|
|dc|Discomfort|20|13|42|
|ti|Tired|19|15|56|
|lo|Lonely|17|12|44|
|ch|Cold/hot|10|8|28|
|sc|Scared|13|6|20|
|dk|Don’t know|134|86|58|

**Data Augmentation**
Our objective was to balance all the 9 classes, by creating new synthetic training samples by adding small perturbations on our initial training set, so that the model is not biased towards any one single class. 
In addition, make our model invariant to those perturbations and enhance its ability to generalize. steps have been followed to extract useful data from the source repository.

The methods used for the process are:
* **Adding white noise:** Added noise with a maximum signal to noise ratio of 0.05 to each channel and generated new samples (see Fig 1.b).
* **Shifting the sound:** Additional audios were generated by shifting the original audio up to 20% of audio’s length. (see Fig 1.c).
* **Stretching the sound:** Generated more audios by speeding up the original audios while keeping the pitch unchanged (see Fig 1.d).

<img src="images/data_processing.png" alt="Data Processing" width="800"/>

## Method
The model described here has given us the highest accuracy out of all the experiments that we have performed. The model training consists of two phases. In the first phase we extract the log Mel-band (MFCC) features. In the second phase we train our deep learning model which consist of convolution, batch norm, pooling, LSTM-RNN, time distributed and attention layers.

**Extract log Mel-band (MFCC) features:** Sound frequencies in real life do not always occur in isolation but tend to considerably overlap with each other. These overlapping sounds can potentially be recognized better with multichannel audio detection techniques. Our input audio consists of two channels.<br>
Below are the various steps involved in extracting the multichannel Spectro-temporal features (spectrogram) from the raw wav audio file and input preprocessing:

* Split the file into fixed sized frames of equal lengths and extract the magnitude spectrum of the audio signals by using short-time Fourier transform (STFT) over 40 ms audio frames of 50% overlap, windowed with Hamming window. The frames for each file range between 287532 and 389632.
* Extracted the log Mel-band energy features for each file using librosa library with a sampling rate of 44,100, nfft = 2048, hop_len = nfft/2. Extracted feature shapes varies from (280, 80) to (381, 80) for different files.
* Since the shape of extracted features varies for different files of varying length, the shape of extracted features for the largest file is taken as the base and all other files are converted to the same shape by padding 0.
* Final feature matrix for each input file is of size (381, 80). Where 381 is the total number of sampled audio frames in the file including padding and each frame consists 80 MFCC features corresponding to two audio channels (40 MFCC features per channel). 
* Normalize the extracted features for each audio file and saved in compressed npz format using NumPy library.

**Deep Learning Model:**

Processing the baby cry is basically a type of pattern recognition. The feature that are invariant to temporal shifts can be learned by Convolutional neural networks (CNN). On the other hand, RNN are effective models to extract the longer-term temporal features in the audio signals. Therefore, we combined and applied these approaches to classify baby cries [1]. We also applied attention model to our network so that only the important frames are selected while the unrelated frames are ignored which allowed us to suppress the background noise, thereby making the system more robust and accurate [2].

The extracted feature matrix is fed to this network and trained for 100 epochs which categorizes each of the 381 frames in a file to a class label. The class label predicted for maximum number of frames is the predicted label for the input audio file. The objective of the model is to classify each frame for the file to its correct class based on the input features for the current frame as well the previous frames (RNN cell state). 

The details of the network layers are as follows:

<table cellspacing="0" cellpadding="0">
  <tr>
    <td>
      &ensp;&ensp;Reshape the (381, 80) input shape to (2, 381, 40) representing (# of audio channels, # for audio frames in the file, Mel-band features). Each audio frame can be thought analogues to image frames in a video file and each channel can be thought of analogues to different color channels in an image.<br><br>
      &ensp;&ensp;2-dimensional Convolutional layer (128 filters) with rectified linear unit/ReLU as activation unit. Kernel size is (5X5). This layer extracts high dimensional, local frequency shift invariant features.<br><br>
      &ensp;&ensp;For regularization to avoid overfitting, we used batch normalization, a pooling layer with filter size 3 and a 30% dropout following the 2D Convolutional layer.<br><br>
      &ensp;&ensp;A permute and reshape layer is being used to reshape the input to be suitable for next LSTM-RNN layer. Input is changed from (# of filters, # of audio frames, # of high-level CNN features) to (# of audio frames, # of high-level CNN features).<br><br>
      &ensp;&ensp;A 128-unit LSTM-RNN layer with tanh activation is being used to captures longer temporal dependencies between the audio frames.<br><br>
      &ensp;&ensp;An attention model is added after the LSTM layer with a tanh Activation Function, Attention Type as Multiplicative and an Attention Width of 32 which learns weights for frames such that the important frames with baby cries will have higher weight values thus increasing the accuracy of the network.<br><br>
      &ensp;&ensp;2 Time-Distributed layers added which are wrappers that applies a standard dense neural network to every temporal slice of an input i.e. each timestep independently, thus enhancing the strength of RNN.<br><br>
      &ensp;&ensp;SoftMax activation is used in final layer to predict the probability of each of the 9-possible outcome. Final output is of shape (381, 9). Each of the audio frame in the input file is assigned a 9-Dimensional probability vector. The majority class across all the frames would be final predicted label for the whole input file.<br><br>
      &ensp;&ensp;The network is trained with back-propagation through time using Adam optimizer and categorical cross-entropy as the loss function.<br><br>
    </td>
    <td>
      <img src="images/model_diag.jpg" alt="Data Processing" width="1500">
    </td>
  </tr>
</table>



