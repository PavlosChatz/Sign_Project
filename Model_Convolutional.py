import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from Utils import data_load_from_directory, load_np_arrays

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3) ) )
    model.add(layers.MaxPooling2D((2, 2))  )
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)) )
    
    model.add(layers.Flatten())  # Fully Connected Layers
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(24, activation = 'softmax'))

    model.summary()
    return model

def createAndTrain(X_train, Y_train, X_validate, Y_validate):
    model = create_model()
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, Y_train, epochs = 10, validation_data = (X_validate, Y_validate) )
    return model

def train_validation_split(X, Y, train_percent):
    size = X.shape[0]
    indices = np.arange(size, dtype = int)
    np.random.RandomState(seed = 0)
    np.random.shuffle(indices)
    border = np.floor(train_percent * size).astype(int) 
    X_train = X[indices[: border], :, :, :]
    Y_train = Y[indices[: border]]
    X_validate = X[indices[border:], :, :, :]
    Y_validate = Y[indices[border: ] ]
    return X_train, Y_train, X_validate, Y_validate

def main():
    #X, Y = data_load_from_directory()
    X, Y = load_np_arrays()

    plt.figure()
    plt.imshow(X[0, :, :, :])
    plt.show()

    X = X / 255

    X_train, Y_train, X_validate, Y_validate = train_validation_split(X, Y, 0.8)

    model = createAndTrain(X_train, Y_train, X_validate, Y_validate)

    return

if __name__ == "__main__":
    main()