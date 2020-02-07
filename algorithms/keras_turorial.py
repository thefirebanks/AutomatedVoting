import keras.backend as K
from keras import Input
from keras.layers import Dense
from keras import Model

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

# Build a model
inputs = Input(shape=(128,))
layer1 = Dense(64, activation='relu')(inputs)
layer2 = Dense(64, activation='relu')(layer1)
predictions = Dense(10, activation='softmax')(layer2)
model = Model(inputs=inputs, outputs=predictions)

# Compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)

    # Return a function
    return loss



# Compile the model
model.compile(optimizer='adam',
              loss=custom_loss(layer1),  # Call the loss function with the selected layer
              metrics=['accuracy'])

# train
model.fit(data, labels)