import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed

def create_model():
        model = Sequential([
        TimeDistributed(Conv1D(filters=256, kernel_size=1,activation='relu'), input_shape=(None,1, 2)),
        TimeDistributed(MaxPooling1D(pool_size=1)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(256, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.50),
        Attention(),
        Dense(32,activation="relu"),
        Dropout(0.20),
        Dense(16,activation="relu"),
        Dropout(0.20),
        Dense(8,activation="relu"),
        Dropout(0.20),
        Dense(11, activation='softmax')
    ])
        optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer,loss=SparseCategoricalCrossentropy(),metrics=['accuracy'])
        return model
    
model = create_model()
model.summary()
