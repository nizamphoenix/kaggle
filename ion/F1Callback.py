import tensorflow as tf
import keras.backend as K
from sklearn.metrics import f1_score

class F1Callback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        pred = np.array([*map(np.argmax,y_pred_val)]).reshape(-1)
        target = self.y_val.reshape(-1)
        score = f1_score(target, pred, average="macro")
        print(f' F1Macro: {score:.5f}')
