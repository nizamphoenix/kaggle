# train
all_predictions = []
from sklearn.model_selection import KFold
import math

kf = KFold(n_splits=5, random_state=42, shuffle=True)

lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * math.pow(drop, math.floor((1+epoch)/0.001)))

for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    model = create_model()
    print( X_tr.shape,y_tr.shape,X_vl.shape,y_vl.shape)
    model.fit(
        X_tr, y_tr, epochs=1, batch_size=64, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[lr_schedule]
    )
    
    print("Done training! Now predicting")
    all_predictions.append(model.predict(X_test))

