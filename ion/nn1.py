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
