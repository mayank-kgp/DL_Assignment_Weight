from zipfile import ZipFile
import numpy as np

'''load your data here'''

class DataLoader(object):
    def __init__(self):
        self.DIR = 'data/'
        self.labels = None
        self.images = None
        pass
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        # One-hot encoding
        self.labels = np.zeros((labels.shape[0],10))
        self.labels[np.arange(labels.shape[0]),labels] = 1
        with ZipFile(image_zip, 'r') as imgzip:
            self.images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(self.labels), 784)
        self.images = self.images.reshape(-1,28,28)
        self.images = self.images/255.0
        # Random Shuffling
        np.random.seed(123)
        p = np.random.permutation(labels.shape[0])
        self.images, self.labels = self.images[p], self.labels[p]
        pass
    # Train-Validation split
    def create_batches(self, split_ratio, batch_size):
        number_of_samples = self.labels.shape[0]
        #Splitting into training and validation
        np.random.seed(123)
        random_indices = np.random.permutation(number_of_samples)
        #Training set
        num_training_samples = int(number_of_samples*split_ratio)
        X_train = self.images[random_indices[:num_training_samples]]
        Y_train = self.labels[random_indices[:num_training_samples]]
        #Create batches
        X = []
        Y = []
        for ndx in range(0, num_training_samples, batch_size):
            if ndx + batch_size > num_training_samples:
                break
            X.append(self.images[ndx:(ndx + batch_size)])
            Y.append(self.labels[ndx:(ndx + batch_size)])
        X = np.array(X)
        Y = np.array(Y)
        #Validation set
        X_val = self.images[random_indices[num_training_samples : ]]
        Y_val = self.labels[random_indices[num_training_samples: ]]
        # X_val = X_val.reshape((1,)+X_val.shape)
        # Y_val = Y_val.reshape((1,)+Y_val.shape)
        return X, Y, X_val, Y_val