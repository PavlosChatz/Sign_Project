import numpy as np 
from PIL import Image
import os

def data_read_from_csv():
    """
    Utility function loading csv data into a np array.
    Path is hardcoded
    Return:
    Array as read from csv with a ',' delimiter
    """

    path = "C:\\Sign_Language_Project\\Dataset"
    data = np.genfromtxt( os.path.join(path, "dataset_greyscale.csv"), delimiter = ',')
    return data#x_train, y_train, x_test, y_test

def data_load_from_directory():
    """
    Returns tuple with X, Y 
    """
    base_path = "C:\\Sign_Language_Project"
    dataset_path = ".\\Dataset\\Greek_Sign_Language_Dataset_Augmented"
    X = np.zeros(shape = (1, 32, 32, 3), dtype = int)
    Y = np.array([], dtype = int)
    count = 0
    for fname in os.listdir(os.path.join(base_path, dataset_path)):
        img = np.array(Image.open(os.path.join(dataset_path, fname) ), ndmin = 4).reshape((1, 32, 32, 3))
        X = np.append(X, img, axis = 0) 
        Y = np.append(Y, np.floor(count / 500 ) ) # Targets (500 images per label)
        count += 1
    #X = X[1:, :, :, :]
    #os.mkdir(os.path.join(base_path, "Dataset\\nparrays"))
    dest = ".\\Dataset\\nparrays"
    np.save(os.path.join(dest, "X.npy"), X) # Save Np Arrays 
    np.save(os.path.join(dest, "Y.npy"), Y) 
    data = (X, Y)
    return data

def load_np_arrays():
    source = ".\\Dataset\\nparrays"
    X = np.load(os.path.join(source, "X.npy"))
    Y = np.load(os.path.join(source, "Y.npy"))
    X = X[1:, :, :, :]
    data = (X, Y)
    return data

if __name__ == "__main__":
    data = data_read_from_csv()
    print(data.shape)
