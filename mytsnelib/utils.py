import csv
import collections.abc as abc
import numpy as np
import matplotlib.pyplot as plt

# def read_csv(route, *, labels_in_first_column=False, num_type=np.int8, skip_start_row=False):
#     """Read a csv file in the given route.

#     Parameters
#     ----------
#     route: str.
#         The route of the csv file to read.

#     has_labels: boolean. default=False.
#         Wether or not the given csv has a labels column at the end.
    
#     index_start: int. default=0.
#         Starting index of the entries to return

#     Returns
#     ---------
#     result: tuple of 2 elements or ndarray of shape (n_samples, n_features)
#         If has_labels is set to True, this is a tuple of 2 elements, where the first is
#         a ndarray of shape (n_samples, n_features) that contains the data in the csv, and a
#         ndarray of shape (1,n_samples) with the labels corresponding to each entry in the first ndarray.
#         Otherwise, it returns the first ndarray of shape (n_samples, n_features)
#     """
#     with open(route, newline='') as csvfile:
#         list_reader = list(csv.reader(csvfile, delimiter=','))
#         array_reader = np.asarray(list_reader[1:], dtype=num_type) if skip_start_row else np.asarray(list_reader, dtype=num_type)
#         if labels_in_first_column:
#             labels = array_reader.T[0]
#             entries = array_reader.T[1:].T
#         else:
#             labels = array_reader.T[-1]
#             entries = array_reader.T[:-1].T
#         return entries, labels
def read_csv(route, *, labels_in_first_column=False, num_type=np.int8, skip_start_row=False):
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
    start_index = 1 if skip_start_row else 0
    if isinstance(route, str):
        data = np.loadtxt(route, num_type, delimiter=',', skiprows=start_index, ndmin=2)
    elif isinstance(route, (abc.Sequence, np.ndarray)):
        data_full = []
        for r in route:
            data_full.append(np.loadtxt(r, num_type, delimiter=',', skiprows=start_index, ndmin=2))
        data = np.vstack(data_full)
    else:
        raise ValueError("Route must be either a str or a sequence of str")
    if labels_in_first_column:
        labels = data.T[0]
        samples = data[:,1:]
    else:
        labels = data.T[-1]
        samples = data[:,:-1]
    return np.asarray(samples), np.asarray(labels)
    # for route in routes:
    #     entry, label = read_csv(route, labels_in_first_column=labels_in_first_column, num_type=num_type,skip_start_row=skip_start_row)
    #     if samples is None:
    #         samples = entry
    #         labels = label
    #     else:
    #     
    #         samples = np.append(samples, entry, axis=0)
    #         labels = np.append(labels, label, axis=0)
    # return np.asarray(samples), np.asarray(labels)


def display_embed(embed, labels, *, title=None):
    is3d = embed.ndim>2
    x = embed.T[0]
    y = embed.T[1]
    if is3d:
        z = embed.T[2]
        for i in range(0, len(embed)):
            plt.plot(x[i], y[i], z[i], marker='o', linestyle='', markersize=5, label=labels[i])
    else:
        for i in range(0, len(embed)):
            plt.plot(x[i], y[i], marker='o', linestyle='', markersize=5, label=labels[i])
    

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), draggable=True, loc='upper right')
    if title is not None:
        plt.title(title)
    plt.show()
