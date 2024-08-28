import csv
import numpy as np
import matplotlib.pyplot as plt
def read_csv(route, *, has_labels=False, samples:int=None, index_start:int=None):
    """Read a csv file in the given route.

    Parameters
    ----------
    route: str.
        The route of the csv file to read.

    has_labels: boolean. default=False.
        Wether or not the given csv has a labels column at the end.

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

        start = 0 if index_start is None else start_index if samples is None else min(index_start, len(array_reader)-samples-1)
        
        n_entries = len(array_reader) if samples is None else min(samples, len(array_reader))
        n_columns = array_reader.shape[1]

        array_reader = array_reader[start:start+n_entries]

        if has_labels:
            labels = array_reader.T[-1]
            entries = array_reader[:,:n_columns-1]
            return entries, labels
        else:
            return array_reader
def display_embed(embedded_data, labels, *, title=None):
    embed_T = embedded_data.T
    x = embed_T[0]
    y = embed_T[1]
    for i in range(0,x.shape[0]):
        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=5, label=labels[i])
    

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), draggable=True)
    if title is not None:
        plt.title(title)
    plt.show()
