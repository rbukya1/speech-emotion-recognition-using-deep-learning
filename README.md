# speech-emotion-recognition-using-deep-learning
Capstone Project - Speech emotion recognition using deep learning 
By
Raja sree Bukya/
Sai Manideep Balmuri/
Meghana Neti

This repository consists of Code, dataset links of Speech Emotion Recognition project.  

Aim of this project is to build a Deep Learning model to recognize emotion of the speaker.  
  
Datasets:
It consists of 3 datasets namely
- Surrey Audio-Visual Expressed Emotion (SAVEE)
- Toronto emotional speech set (TESS)
- Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
  
combined for modeling.<br>The Datasets file consists of dataset briefing as well as website link to download datsets.
  
The project consists of 3 parts.  
1. Part-I constitutes of Loading and combining data from 3 datasets into a dataframe which contains 2 columns, path of the file and label(emotion) for that file, and visualizing the audio by utilizing waveplots as well as spectograms.
2. Part-II focuses on processing the data for modeling, this includes feature extraction to convert audio files into set of numericals required to perform modeling.
   This feature extraction is done using 4 different methods: MFCC (Mel-Frequency Cepstral Coefficients), Chroma, ZCR (Zero Crossing Rate), RMSE (root mean squared error).
3. Part-III In this section, we used 3 deep learning models
   - LSTM (Long Short-Term Memory)
   - CNN (Convolutional Neural Network)
   - GRU (Gated Recurrent Unit)  

and achieved good accuracy
  
Final python file is in directory Phase-3 along with saved CNN model and application file deployed using streamlit.


