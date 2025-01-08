import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Sequential
from tensorflow.keras.layers import Reshape

# Load the audio file using librosa
def load_audio(filename, sr=16000):
    audio, sr = librosa.load(filename, sr=sr)
    return audio, sr

# Preprocess the audio data
def preprocess_audio(audio, sr, n_mfcc=40, n_mels=80):
    audio = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    audio = np.expand_dims(audio, axis=-1)
    return audio

# Build the speech-to-text neural network model
def build_model(input_shape, output_shape, rnn_type=LSTM, dense_units=128, dropout=0.1):
    model = Sequential()
    input_tensor = Input(shape=input_shape)
    reshaped_input = Reshape((input_shape[0], input_shape[1], 1))(input_tensor)
    if rnn_type == 'lstm':
        rnn = LSTM
    elif rnn_type == 'gru':
        rnn = GRU
    else:
        raise ValueError("Invalid RNN type")
    model.add(Bidirectional(rnn(dense_units, return_sequences=True)))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(rnn_type(dense_units, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(output_shape, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function to extract text from audio
def extract_text(filename, model):
    # Load the audio file and preprocess the data
    audio, sr = load_audio(filename)
    audio = preprocess_audio(audio, sr)

    # Predict the transcript using the trained model
    prediction = model.predict(audio)
    prediction = np.argmax(prediction, axis=-1)

    # Convert the predicted transcript to text and write to a file
    transcript = ''
    for i in range(prediction.shape[0]):
        transcript += chr(prediction[i] + 97)
    with open('transcript.txt', 'w') as file:
        file.write(transcript)

if __name__ == '__main__':
    input_shape = (40, 1)
    output_shape = 26

    model = build_model(input_shape, output_shape)
    extract_text('audio.mp3', model)
