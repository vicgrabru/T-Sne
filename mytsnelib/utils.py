import csv
import numpy as np
import matplotlib.pyplot as plt

import struct
from array import array


class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test) 



def read_csv(route, *, has_labels=False, index_start=0, samples=None):
    """Read a csv file in the given route.

    Parameters
    ----------
    route: str.
        The route of the csv file to read.

    has_labels: boolean. default=False.
        Wether or not the given csv has a labels column at the end.
    
    index_start: int. default=0.
        Starting index of the entries to return

    Returns
    ---------
    result: tuple of 2 elements or ndarray of shape (n_samples, n_features)
        If has_labels is set to True, this is a tuple of 2 elements, where the first is
        a ndarray of shape (n_samples, n_features) that contains the data in the csv, and a
        ndarray of shape (1,n_samples) with the labels corresponding to each entry in the first ndarray.
        Otherwise, it returns the first ndarray of shape (n_samples, n_features)
    """
    with open(route, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        list_reader = list(reader)
        array_reader = np.asarray(list_reader)

        max_index = len(array_reader)-1
        if samples is not None:
            max_index -= samples

        start = index_start % max_index
        end = start + samples + 1

        data = array_reader[start:end]

        if has_labels:
            labels = data.T[-1]
            entries = data.T[:-1].T
            return entries, labels
        else:
            return data



def display_embed(embed, labels, *, title=None):
    x = embed.T[0]
    y = embed.T[1]
    for i in range(0,x.shape[0]):
        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=5, label=labels[i])
    

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), draggable=True)
    if title is not None:
        plt.title(title)
    plt.show()
