import librosa
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import requests
import streamlit as st
from tensorflow.keras.models import load_model

github_raw_url = 'https://github.com/Sbalmur1/Emotion-Recognition-App/raw/main/CNN_model.h5'  # Updated raw URL
local_filename = 'CNN_model.h5'

response = requests.get(github_raw_url)

if response.status_code == 200:
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    print('File downloaded successfully')
else:
    print("Failed to fetch file from GitHub")

# load model
model = load_model(local_filename)

# summarize model.
model.summary()


# compute MFCCs for each audio file
def extract_mfcc(y, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def extract_chroma(y, sr):
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    return chroma

def extract_zcr(y):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    return zcr

def extract_rmse(y):
    D = librosa.stft(y)
    y_reconstructed = librosa.istft(D, length=len(y))
    rmse = np.sqrt(np.mean((y - y_reconstructed) ** 2))
    return rmse





# Function to get emotion prediction for a single audio file
def predict_emotion(audio_file):
  #Extracting features
  y, sr = librosa.load(audio_file, duration=3, offset=0.5)
  mfcc = extract_mfcc(y, sr)
  chroma = extract_chroma(y, sr)
  zcr = extract_zcr(y)
  rmse = np.array([extract_rmse(y)])

  # Create DataFrames for each feature
  mfcc = pd.DataFrame(mfcc.reshape(1, -1))  # Reshape to 1 row and as many columns as needed
  chroma = pd.DataFrame(chroma.reshape(1, -1))
  zcr = pd.DataFrame(zcr.reshape(1, -1))
  rmse = pd.DataFrame(rmse.reshape(1, -1))

  # Concatenate all feature sets into a single DataFrame
  combined = pd.concat([mfcc, chroma, zcr, rmse], axis=1)
    
  # Get prediction from the model
  predicted_probabilities = model.predict(combined)
    
  # Get the index of the emotion with highest probability
  predicted_label_index = np.argmax(predicted_probabilities)
    
  # Map index to emotion label (assuming index 0 is 'angry', index 1 is 'happy', etc.)
  emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
  predicted_emotion = emotions[predicted_label_index]

  return predicted_emotion



def main():
    
    
    # giving a title
    st.title('Emotion Prediction Web App')

    # File upload widget
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    
    predicted_emotion = ''
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format = "audio/wav")

    if st.button("Predict Emotion"):
        # Make emotion prediction
        predicted_emotion = predict_emotion(uploaded_file)
    
    st.success(f"Predicted Emotion: {predicted_emotion}")



if __name__ == "__main__":
    main()