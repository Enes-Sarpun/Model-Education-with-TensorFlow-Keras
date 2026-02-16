# Import necessary libraries;
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report
import warnings 

# Ignore warnings for cleaner output;
warnings.filterwarnings('ignore')

## Data and Preprocessing -> Normalization and One-Hot Encoding;
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

# Little Visualization of Data;
class_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
fig,axes = plt.subplots(2,5,figsize=(10,5))
for i,ax in enumerate(axes.flat):
    ax.imshow(x_train[i])
    ax.set_title(class_name[y_train[i][0]])
    ax.axis('off')
plt.show()

# Normalization;
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0 

# One-Hot Encoding;
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)


## Data Augmentation -> ImageDataGenerator;
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

datagen.fit(x_train)
print("-"*25)
print(x_train.shape[1:])
print("-"*25)

## Model Building -> Compile and Train Model;
Model = Sequential()
Model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=x_train.shape[1:]))
Model.add(Conv2D(32,(3,3),activation='relu'))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Dropout(0.25))

Model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
Model.add(Conv2D(64,(3,3),activation='relu'))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Dropout(0.25))

Model.add(Flatten())
Model.add(Dense(512,activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(10,activation='softmax'))

Model.summary()

# Compile Model;
Model.compile(optimizer=RMSprop(learning_rate=0.0001,decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])

history = Model.fit(datagen.flow(x_train,y_train,batch_size=64),epochs=20,validation_data=(x_test,y_test))

## Evaluation -> Test and Evaluate Model;
y_pred = Model.predict(x_test)
y_pred_class = np.argmax(y_pred,axis=1) # Get class with highest probability!
y_true = np.argmax(y_test,axis=1)

# Classfication Report;
print(classification_report(y_true,y_pred_class,target_names=class_name))

# Loss Plot;
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Accuracy Plot;
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Finished.