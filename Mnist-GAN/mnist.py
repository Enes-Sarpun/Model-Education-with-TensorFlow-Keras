# Load the datasets;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Model Normalization and Preprocessing;
(x_train,_),(_,_) = mnist.load_data()

# Normalization;
x_train = x_train / 255.0

# (28,28) to (28,28,1);
x_train = np.expand_dims(x_train, axis=-1)

# Describe Discriminator and Generator;
z_dim = 100
def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(28,28,1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten()) # Image was converted vectoral shape! 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5), metrics=['accuracy'])
    return model

def build_generator():
    model = Sequential()
    model.add(Dense(7*7*128, input_dim=z_dim)) # noise vector convert to high dimensional space!
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same'))
    return model

# Build the GAN Model;
def Gan_Model(generator, Discriminator):
    Discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(Discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

discriminator = build_discriminator()
generator = build_generator()
gan = Gan_Model(generator, discriminator)
print(generator.summary())

# Train and Evaluate the GAN;
epochs = 3000
batch_size = 64
half_batch = int(batch_size / 2)

# Train Discriminator;
for epoch in tqdm(range(epochs)):
    idx = np.random.randint(0, x_train.shape[0], half_batch) # Batch_size = 32
    real_imgs = x_train[idx]
    real_labels = np.zeros((half_batch, 1))

    noise = np.random.normal(0, 1, (half_batch, z_dim))
    fake_imgs = generator.predict(noise)
    fake_labels = np.ones((half_batch, 1))

    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Epoch for Generator;
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    valid_y = np.zeros((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)

for epoch in tqdm(range(epochs)):
    if epoch % 1000 == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

# Visualize the Results;
def plot_generated_images(generator, epoch, examples=10, dim=(1,10), figsize=(10,1)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, z_dim])
    generated_images = generator.predict(noise, verbose=0)
    generated_images = 0.5*generated_images + 0.5

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_generated_images(generator, epoch)


# Finished.