# Import Necessary Libraries;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist as fashion
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim

# Load Datasets, Normalizaton and Reshape;
(x_train, _), (x_test, _) = fashion.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Visualize Sample Images;
plt.subplot(1, 2, 1)
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
plt.show()
 
# Reshape Data for Model Input;
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # Image was converted (28,28) to vector 784.
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Build the Autoencoder Model;
input_dim = x_train.shape[1] # 784
encoding_dim = 64

# Encoder Network; --> We will compress the image to 64 dimensions;
input_image = Input(shape=(input_dim,))
encoded= Dense(512, activation='relu')(input_image)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128,activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded) 

# Decoder Network; --> We will reconstruct the image back to original dimensions;
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Combine Encoder and Decoder into Autoencoder Model;
autoencoder = Model(input_image, decoded)
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
autoencoder.summary()

# Train the Autoencoder;
history = autoencoder.fit(x_train, x_train, epochs=20, batch_size=64, shuffle=True, validation_data=(x_test, x_test),verbose = 1)

# Encoder and Decoder; 

encoder = Model(input_image, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoded = autoencoder.layers[-4](encoded_input)
decoded = autoencoder.layers[-3](decoded)
decoded = autoencoder.layers[-2](decoded)
decoded = autoencoder.layers[-1](decoded)
decoder = Model(encoded_input, decoded)

# Encode and Decode Some Test Images;
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # Number of images to display;
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original;
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    # Display reconstruction;
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()

# SSIM; 
def compute_ssim(original,reconstructed):
    original = original.reshape(28,28)
    reconstructed = reconstructed.reshape(28,28)
    return ssim(original, reconstructed,data_range=1)

ssim_scores = []
for i in range(len(x_test)):
    original_image = x_test[i]
    reconstructed_image = decoded_imgs[i]
    score = compute_ssim(original_image, reconstructed_image)
    ssim_scores.append(score)

average_ssim = np.mean(ssim_scores)
print(f'Average SSIM between original and reconstructed images: {average_ssim:.4f}')

# Finished.