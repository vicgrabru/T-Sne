import csv
import numpy as np
import matplotlib.pyplot as plt

import struct
from array import array



def _read_mnist(data_route, labels_route):
        labels = []
        with open(labels_route, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        
        with open(data_route, 'rb') as file:
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
        return np.asarray(images, dtype=np.int8), np.asarray(labels, dtype=np.int8)
def read_mnist(training_data_route, training_labels_route, test_data_route, test_labels_route, dtype=np.int8):
        data_train, labels_train = _read_mnist(training_data_route, training_labels_route, type=type)
        data_test, labels_test = _read_mnist(test_data_route, test_labels_route)
        return (data_train, labels_train),(data_test, labels_test) 


def read_csv(route, *, has_labels=False, labels_in_first_column=False):
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

        

        if has_labels:
            if labels_in_first_column:
                labels = array_reader.T[0]
                entries = array_reader.T[1:].T
            else:
                labels = array_reader.T[-1]
                entries = array_reader.T[:-1].T
            return entries, labels
        else:
            return array_reader



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
