#->  Load Data;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer, Flatten
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings("ignore")

# Load Iris Dataset;
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# One-Hot Encoding;
y_onehot = LabelBinarizer().fit_transform(y)

# Standardization;
x_scaler = StandardScaler().fit_transform(X)

# Train-Test Split;
(X_train, X_test, y_train, y_test) = train_test_split(x_scaler, y_onehot, test_size=0.2, random_state=42)

#-> RBF Layer;
class RBFLayer(Layer):
    def __init__(self,units,gamma,**kwargs):
        """
        Constructor, It's required to start for the layers.
        """
        super(RBFLayer,self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)
    
    def build(self,input_shape):
        """
        Create the layer weights. This method is called once the layer is called the first time.
        """
        self.mu = self.add_weight(name='mu',
                                       shape=(self.units,input_shape[1]), # Shape = It desribe the weights dimensions
                                       initializer='uniform', # Distributes the starting weights evenly.
                                       trainable=True) # Weights are trainable.
        super(RBFLayer,self).build(input_shape) # Be sure to call this at the end.

    def call(self,inputs):
        """
        Forward Propagation. It takes input and gives output.        
        """
        diff = K.expand_dims(inputs,axis=1) - self.mu
        l2 = K.sum(K.pow(diff,2),axis=-1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self,input_shape):
        """
        Compute the output shape of the layer.
        """
        return (input_shape[0],self.units)

#-> Model Definition;
def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(4,))) # Input Layer
    model.add(RBFLayer(10,0.5)) # RBF Layer with 10 neurons and gamma=0.5
    model.add(Dense(3,activation='softmax')) # Output Layer
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = build_model()
model.summary()

history = model.fit(X_train,y_train,validation_split=0.3,epochs=100,batch_size=4,verbose=1)

# -> Evaluation;
loss,accuracy = model.evaluate(X_test,y_test,verbose=0)
print("Test Loss:",loss,"Test Accuracy:",accuracy)

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()    
plt.show()


# Finished.