import tensorflow.keras as keras
import numpy as np

from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential

def printDeviceCapabilities():
    print(device_lib.list_local_devices())

def readPretoBranco():
    file = open('pretobranco_x.npy', 'rb')
    db_x = np.load(file)

    file = open('pretobranco_y.npy', 'rb')
    db_y = np.load(file)

    return db_x, db_y

# printDeviceCapabilities()

print("reading database")
db_x, db_y = readPretoBranco()
print("db_x.shape: ", db_x.shape)
print("db_y.shape: ", db_y.shape)