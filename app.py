import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
from keras.models import load_model
import io

# Load the trained model
model = load_model('C:/Users/Divam/speech_confidence_model.h5')

# Define feature extraction function
def extract_features(y, sr, max_length=216):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < max_length:
        pad_width = max_length - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_length]
    return log_mel

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.pexels.com/photos/3772511/pexels-photo-3772511.jpeg?auto=compress&cs=tinysrgb&w=600'); 
        background-size: contain;
        background-repeat:no-repeat;
        background-position: center;
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }
    .content {
        text-align: left;
    }
    .stMarkdown {
        color: black;
        
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .interviewSection {
        margin-top: 40px;
        text-align: left;
    }
    .interviewSection ul {
        list-style-type: none;
        padding: 0;
    }
    .interviewSection li {
        margin-bottom: 10px;
    }
    h2 {
        color: #000000;
        font-weight: bold;
    }
    h1 {
        color: #000000;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.title('KNOW HOW CONFIDENT YOU ARE!')

    st.write('<div class="content stMarkdown">Record your voice or upload an audio file to predict whether you sound confident or not.</div>', unsafe_allow_html=True)

    option = st.radio('Choose an option:', ('Record', 'Upload'))

    if option == 'Record':
        if st.button('Record'):
            duration = 10  # seconds
            fs = 22050  # sample rate

            try:
                st.write('Recording...')
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()
                st.write('Recording finished.')

                # Extract features from the recorded audio
                y = recording.flatten()
                log_mel = extract_features(y, sr=fs)
                log_mel = np.expand_dims(log_mel, axis=-1)
                log_mel = np.expand_dims(log_mel, axis=0)

                # Predict confidence
                prediction = model.predict(log_mel)
                confidence_label = np.argmax(prediction, axis=1)[0]
                confidence_percentage = np.max(prediction) * 100

                label_mapping = {0: 'non confident', 1: 'confident'}
                st.write(f'Prediction: {label_mapping[confidence_label]}')
                st.write(f'Your Confidence percentage is: {confidence_percentage:.2f}%')

            except Exception as e:
                st.error(f"An error occurred during recording: {e}")

    elif option == 'Upload':
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

        if uploaded_file is not None:
            try:
                # Load the uploaded audio file
                y, sr = librosa.load(io.BytesIO(uploaded_file.read()), sr=22050)
                st.audio(uploaded_file, format='audio/wav')

                # Extract features from the uploaded audio
                log_mel = extract_features(y, sr=sr)
                log_mel = np.expand_dims(log_mel, axis=-1)
                log_mel = np.expand_dims(log_mel, axis=0)

                if st.button('Predict'):
                    # Predict confidence
                    prediction = model.predict(log_mel)
                    confidence_label = np.argmax(prediction, axis=1)[0]
                    confidence_percentage = np.max(prediction) * 100

                    label_mapping = {0: 'non confident', 1: 'confident'}
                    st.write(f'Prediction: {label_mapping[confidence_label]}')
                    st.write(f'Confidence: {confidence_percentage:.2f}%')

            except Exception as e:
                st.error(f"An error occurred during file upload or prediction: {e}")

    st.markdown("""
    <div class="interviewSection">
        <h2>INTERVIEW PRACTICE</h2>
        <p>If you forgot what to speak, try answering these common interview questions:</p>
        <ul>
            <li><b>Question 1:</b> Tell me about yourself.</li>
            <li><b>Question 2:</b> Why do you want to work here?</li>
            <li><b>Question 3:</b> What are your strengths and weaknesses?</li>
            <li><b>Question 4:</b> Describe a challenging situation you faced and how you handled it.</li>
            <li><b>Question 5:</b> Where do you see yourself in 5 years?</li>
        </ul>
        <p>Record your answer to these questions:</p>
    </div>
    """, unsafe_allow_html=True)

    interview_option = st.radio('Choose an interview question to answer:', 
                                ('Tell me about yourself.', 
                                'Why do you want to work here?', 
                                'What are your strengths and weaknesses?', 
                                'Describe a challenging situation you faced and how you handled it.', 
                                'Where do you see yourself in 5 years?'))

    if st.button('Record Answer'):
        duration = 10  # seconds
        fs = 22050  # sample rate

        try:
            st.write('Recording...')
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            st.write('Recording finished.')

            # Extract features from the recorded audio
            y = recording.flatten()
            log_mel = extract_features(y, sr=fs)
            log_mel = np.expand_dims(log_mel, axis=-1)
            log_mel = np.expand_dims(log_mel, axis=0)

            # Predict confidence
            prediction = model.predict(log_mel)
            confidence_label = np.argmax(prediction, axis=1)[0]
            confidence_percentage = np.max(prediction) * 100

            label_mapping = {0: 'non confident', 1: 'confident'}
            st.write(f'Prediction: {label_mapping[confidence_label]}')
            st.write(f'Your Confidence percentage is: {confidence_percentage:.2f}%')

        except Exception as e:
            st.error(f"An error occurred during recording: {e}")
