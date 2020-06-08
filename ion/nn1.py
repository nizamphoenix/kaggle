%%time
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>MODEL-1
n_classes = 11   
model = Sequential([
Conv1D(filters=32, kernel_size=1, kernel_initializer='he_normal',activation='relu', input_shape=(1,1)),
BatchNormalization(),
# model.add(MaxPool1D(pool_size=2))
Dropout(0.25),
Conv1D(filters=64, kernel_size=1, activation='relu'),
BatchNormalization(),
# model.add(MaxPool1D(pool_size=2))
Dropout(0.50),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.30),
Dense(10, activation='relu'),
Dropout(0.50),
Dense(n_classes, activation='softmax')])
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
#--
history = model.fit(X_train,y_train)
#-------------------
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs

#------------------------------------------------
# Plot MAE and Loss
#------------------------------------------------
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()

epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]

#------------------------------------------------
# Plot Zoomed MAE and Loss
#------------------------------------------------
plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()
