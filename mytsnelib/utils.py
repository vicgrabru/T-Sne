import csv
import numpy as np
def read_csv(route, has_labels=False):
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
        n_entries = array_reader.shape[0]
        n_columns = array_reader.shape[1]
        if has_labels:
            labels = array_reader.T[-1]
            entries = array_reader[:n_entries, :n_columns-1]
            return entries, labels
        else:
            entries = array_reader[:n_entries, :n_columns]
            return entries
        
