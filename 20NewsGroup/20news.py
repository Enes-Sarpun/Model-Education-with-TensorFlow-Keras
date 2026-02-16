# Import Necessary Libraries;
from xml.parsers.expat import model
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings('ignore')

##  Load data and Preprocessing;
new_group = fetch_20newsgroups(subset='all') # We loaded the data and separated it into X-Y files.
X = new_group.data
Y = new_group.target

# Tokienization;
tokenizer = Tokenizer(num_words=10000) 
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X) # Convert text to sequences.
X_pad = pad_sequences(X_seq, maxlen=100) # It pad the sequences to make them of equal length.

# Encoder;
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y) # Ticket the labels to numerical values.

# Train Test Split;

X_train, X_test, Y_train, Y_test = train_test_split(X_pad, Y_encoded, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

## Model Building;

def f1_Score(y_true, y_pred): # Metric Function;
    y_pred = K.argmax(y_pred, axis=1)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

def Build_Model():
    model = Sequential()

    model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy', f1_Score])
    model.summary()
    
    return model

model = Build_Model()

# Train LSTM;
history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)])

def Evaluate_Model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
Evaluate_Model(model, X_test, Y_test)

plt.subplots(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplots(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')   
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Finished.