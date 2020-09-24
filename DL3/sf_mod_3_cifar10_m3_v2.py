

# Импорты



import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as M
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import numpy as np
print(tf.__version__)
print(keras.__version__)



physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""# Загружаем данные"""

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Трейн:", x_train.shape, y_train.shape)
print("Тест:", x_test.shape, y_test.shape)

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]



# нормализуем входы
x_train2 = x_train.astype('float32') / 255 - 0.5
x_test2 = x_test.astype('float32') / 255 - 0.5

# конвертируем метки в np.array (?, NUM_CLASSES)
y_train2 = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test2 = keras.utils.to_categorical(y_test, NUM_CLASSES)

"""# Задаем *дефолтную* архитектуру сети"""

# слои, которые нам пригодятся
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def make_default_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', ))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', ))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', ))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    return model

K.clear_session()
model = make_default_model()
model.summary()

"""# Тренируем модель"""

def train_model(make_model_func=make_default_model, optimizer="adam"):
  BATCH_SIZE = 32
  EPOCHS = 10

  K.clear_session()
  model = make_model_func()

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy']
  )

  model.fit(
      x_train2, y_train2,  # нормализованные данные
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      validation_data=(x_test2, y_test2),
      shuffle=False
  )
  
  return model


# учим дефолтную архитектуру
train_model()

'''def make_sigmoid_model():
    ...

# учим sigmoid
train_model(make_sigmoid_model)

# учим sgd
train_model(optimizer="sgd")

def make_bn_model():
    ...

# учим bn
train_model(make_bn_model)

def make_sigmoid_bn_model():
    ...

# учим sigmoid + bn
train_model(make_sigmoid_bn_model)
'''

