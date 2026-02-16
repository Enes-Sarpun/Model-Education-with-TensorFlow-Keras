# Import Necessary Libraries;
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib as path
import os
import warnings
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Rescaling, RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")


# Load Data, Preprocessing, Data Augmentation, Resize, Rescale, Train-Test Split;
data_file = 'Drug Vision/Data Combined'
image_dir = path.Path(data_file)

file_paths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_paths))

file_paths = pd.Series(file_paths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

df = pd.concat([file_paths, labels], axis=1)

random_index = np.random.randint(0,len(df),25)
fig,ax = plt.subplots(5,5, figsize=(10,10))
for i,ax in enumerate(ax.flat):
    img = plt.imread(df.Filepath[random_index[i]])
    ax.imshow(img)
    ax.set_title(df.Label[random_index[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df, 
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',   
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',   
    class_mode='categorical',
    batch_size=64,
    shuffle=False
)

resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224), # Resize to 224x224.
    Rescaling(1./255) # Normalization.
])

# MobileNetV2; # We will not use the mobile net classification layers (trainable=False)!
base_model = MobileNetV2(input_shape=(224,224,3),
                         include_top=False, # Exclude the top classification layer.
                         weights='imagenet')
base_model.trainable = False # Freeze the base model.

model_checkpoint = ModelCheckpoint('checkpoint.weights.h5',
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=1)

early_stop = EarlyStopping(monitor='val_loss',
                            patience=5,
                            restore_best_weights=True,
                            verbose=1)

# Build Model;
inputs = base_model.input
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x) 
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile Model;

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model;
history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    validation_data=val_images,
    validation_steps=len(val_images),
    epochs=15,
    callbacks=[model_checkpoint, early_stop]
)

# Predict Model;

loss, acc = model.evaluate(test_images, verbose=1)
print(f'Test Loss: {loss}, Test Accuracy: {acc}')

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.grid(True)
plt.title('Loss over epochs')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.grid(True)
plt.title('Accuracy over epochs')
plt.show()

pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)

labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())

pred_labels = [labels[k] for k in pred]

random_index = np.random.randint(0, len(test_df), 25)
fig, ax = plt.subplots(5,5, figsize=(10,10))
for i,ax in enumerate(ax.flat):
    img = plt.imread(test_df.Filepath[random_index[i]])
    ax.imshow(img)  
    if pred_labels[random_index[i]] == test_df.Label[random_index[i]]:
        ax.set_title(f"Pred: {pred_labels[random_index[i]]}\nTrue: {test_df.Label[random_index[i]]}", color='green')
    else:
        ax.set_title(f"Pred: {pred_labels[random_index[i]]}\nTrue: {test_df.Label[random_index[i]]}", color='red')
    ax.axis('off')
plt.tight_layout()
plt.show()




# Finished.