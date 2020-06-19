import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
import math


def train(X_train,y_train,BATCH_SIZE):
    K = 3
    EPOCHS = 2
    oof_predictions = []
    kf = KFold(n_splits=K, random_state=1, shuffle=True)
    lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * math.pow(0.001, math.floor((1+epoch)/3.0)))
    early_stopping = EarlyStopping(
        min_delta = 0.001,
        monitor='val_auc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True
    )
    test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE))
    #type(test_dataset):tensorflow.python.data.ops.dataset_ops.PrefetchDataset
    
    for ind, (tr, val) in enumerate(kf.split(X_train)):
        X_tr = X_train[tr]
        y_tr = y_train[tr]
        X_vl = X_train[val]
        y_vl = y_train[val]
        print(X_tr.shape,y_tr.shape,X_vl.shape,y_vl.shape)
        
        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((X_tr,y_tr))
            .repeat()
            .shuffle(2048)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )

        valid_dataset = (
            tf.data.Dataset
            .from_tensor_slices((X_vl,y_vl))
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
        )
        
        n_steps = X_tr.shape[0] // BATCH_SIZE
        train_history = model.fit(
                        train_dataset,
                        steps_per_epoch=n_steps,
                        validation_data=valid_dataset,
                        epochs=EPOCHS,
                        verbose=True, 
                        callbacks=[lr_schedule]
        )
      
        print("Done training! Now predicting")
        oof_predictions.append(model.predict(test_dataset, verbose=1))
    return oof_predictions
