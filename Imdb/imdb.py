# Import Necessary Libraries, padding;
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_curve, auc
import kerastuner as kt 
from kerastuner.tuners import RandomSearch
import warnings
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
maxlen = 500
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build RNN Model;
def BuildModel(hp):
    # Embedding Layer turn words into vectors!
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=hp.Int('embedding_output_dim',min_value=32, max_value=128, step=32), input_length=maxlen))
    model.add(SimpleRNN(units=hp.Int('rnn_units', min_value=32, max_value=128, step=32), activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=hp.Choice('optimizer',['adam','rmsprop']), loss='binary_crossentropy', metrics=['accuracy','AUC'])
    return model

# Hyperparameters and Train model;
tuner = RandomSearch(BuildModel,
                     objective='val_loss', # The best model will minimize validation loss!
                     max_trials=2, # It's trying 2 different combinations of hyperparameters!
                     executions_per_trial=1, # Once per for each combination!
                     directory='imdb_rnn_tuning', # Directory to save results!
                     project_name='imdb_sentiment_analysis')

tuner.search(x_train, y_train, epochs=2, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])

# Evaluate Model;
best_model = tuner.get_best_models(num_models=1)[0]

loss,acc,auc = best_model.evaluate(x_test,y_test)
print(f'Test Loss: {loss}, Test Accuracy: {acc}, Test AUC: {auc}')

y_pred_prob = best_model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# ROC Curve and AUC;
fpr, tpr, _ = roc_curve(y_test, y_pred_prob) # False Positive Rate, True Positive Rate
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc}')

# Visualize ROC Curve;
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Finished.