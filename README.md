# Speech Emotion Recognition using Deep Learning

## Capstone Project

This repository consists of code and dataset links for a Speech Emotion Recognition project.

### Aim
The aim of this project is to build a Deep Learning model to recognize the emotion of the speaker.

### Datasets
The project utilizes three datasets:
- Surrey Audio-Visual Expressed Emotion (SAVEE)
- Toronto emotional speech set (TESS)
- Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

Combined for modeling. The Datasets file includes dataset briefing as well as website links to download datasets.

### Project Overview
The project consists of three main parts:

1. **Data Loading and Visualization**
   - Combining data from the three datasets into a dataframe with two columns: file path and emotion label.
   - Visualizing the audio using waveplots and spectrograms.

2. **Data Preprocessing**
   Feature extraction methods include:
     - MFCC (Mel-Frequency Cepstral Coefficients)
     - Chroma
     - ZCR (Zero Crossing Rate)
     - RMSE (Root Mean Squared Error)

3. **Modeling**
   Utilizing three deep learning models:
     - LSTM (Long Short-Term Memory)
     - CNN (Convolutional Neural Network)
     - GRU (Gated Recurrent Unit)
   Achieving good accuracy.

### Final Deliverables
The final Python file is located in the Phase-3 directory along with the saved CNN model and an application file deployed using Streamlit.

