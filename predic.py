from prepare_data import SPLITED_DATA
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
mapping = {1:'pickachu', 0:'none'}


X_train, X_valid, y_train, y_valid = SPLITED_DATA

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer
X_train = base_model.predict(X_train) #extract features
X_valid = base_model.predict(X_valid)

X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(90, 7*7*512)

train = X_train/X_train.max()      # centering the data (0..1)
X_valid = X_valid/X_train.max()

model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(3, activation='softmax'))    # output layer

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

model.save('model.h5')

#model = keras.models.load_model('model.h5')