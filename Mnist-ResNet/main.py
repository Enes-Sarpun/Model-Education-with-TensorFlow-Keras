# Import Necessary Libraries;
import tensorflow as tf
import os
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras_tuner import HyperModel, RandomSearch

# Load Data, ReShape/Scale, One-Hot Encode Labels;
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Residual Block Definition;
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# ResNetModel;
class ResNetModel(HyperModel):
    def build(self,hp):
        inputs=Input(shape=(28,28,1))
        x=Conv2D(filters=hp.Int('initial_filters',min_value=32,max_value=128,step=32), kernel_size=3, strides=1, padding='same')(inputs)
        x=BatchNormalization()(x)

        for i in range(hp.Int('num_res_blocks',min_value=1,max_value=3,step=1)):
            x=residual_block(x, filters=hp.Int(f'filters_block_{i}',min_value=32,max_value=128,step=32))
        
        # Classification;
        x=Flatten()(x)
        x=Dense(128, activation='relu')(x)
        outputs=Dense(10, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate',min_value=1e-4,max_value=1e-2, sampling='LOG')),
                       loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
# Model;
tuning_dir = os.path.join(os.environ.get('TEMP', 'C:\\Temp'), 'keras_tuning')
tuner = RandomSearch(
    ResNetModel(),
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory=tuning_dir,
    project_name='resnet_fm'
)

tuner.search(x_train, y_train, epochs=4, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")


# Finished.