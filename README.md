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