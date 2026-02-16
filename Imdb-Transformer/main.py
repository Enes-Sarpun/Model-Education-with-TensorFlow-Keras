# Import Necessary Libraries;
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Max Feature and Len;
max_features = 10000
maxlen = 100

# Load data and preprocess;
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure uniform input length;
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

word_index = imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = '<PAD>'
index_word[1] = '<START>'
index_word[2] = '<UNK>'
index_word[3] = '<UNUSED>'

# Look at the Texts;
def decode_review(text):

    return ' '.join([index_word.get(i, '?') for i in text])

random_review = np.random.choice(len(x_train), size=3, replace=False)
for review in random_review:
    print(decode_review(x_train[review]))
    print("-"*25)

# Transformer Block Definition;
class transformer_block(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_head, ff_dim, dropout_rate=0.3):
        super(transformer_block, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_head, key_dim=embed_dim // num_head)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)    
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attention = self.attention(inputs, inputs)
        inputs = self.norm1(inputs + self.dropout1(attention, training=training))
        ffn_output = self.ffn(inputs)
        return self.norm2(inputs + self.dropout2(ffn_output, training=training))    

# Model Definition;
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=embed_size)
        self.transformer_blocks = [transformer_block(embed_size, num_heads, embed_size*4, dropout_rate) for _ in range(num_layers)]
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs, training):
        x = self.embedding(inputs)
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.fc(x)

# Hyperparameters;
num_layers = 4
embed_size = 64
num_heads = 4
input_dim = max_features
output_dim = 1
dropout_rate = 0.1

model = TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate)

# Build the model with a dummy input to initialize all layers
dummy_input = np.zeros((1, maxlen), dtype=np.int32)
_ = model(dummy_input, training=False)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training;
history = model.fit(x_train, y_train, batch_size=128, epochs=3, validation_data=(x_test, y_test))

# Evaluation;
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Give the users text;
def predict_review(text):
    encoded_text = [word_index.get(word,0) for word in text.lower().split()]
    padded_text = sequence.pad_sequences([encoded_text], maxlen=maxlen)
    prediction = model.predict(padded_text)
    return 'Positive' if prediction > 0.5 else 'Negative'

# User Input;
user_input = input("Enter a movie review: ")
result = predict_review(user_input)
print(f"The review is predicted to be: {result}")



# Finished.