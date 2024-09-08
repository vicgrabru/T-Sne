import csv
import collections.abc as abc
import numpy as np
import matplotlib.pyplot as plt
import gc


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
        del data_full
    else:
        raise ValueError("Route must be either a str or a sequence of str")
    if labels_in_first_column:
        labels = data.T[0]
        samples = data[:,1:]
    else:
        labels = data.T[-1]
        samples = data[:,:-1]
    del start_index,data; gc.collect()
    return np.asarray(samples), np.asarray(labels)


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


def print_tiempo(t, metodo="Mio", n_digits_ms=6):
    t_exact = np.floor(t)
    tS = str(t_exact%60 + (t - t_exact))[:n_digits_ms+2]
    tM = int(np.floor(t_exact/60)%60)
    tH = int(np.floor(t_exact/3600))
    print("=====================================================")
    print(metodo + " finished")
    if t_exact>60:
        if t_exact<3600: # <1h
            print("Execution time (min:sec): {}:{}".format(tM,tS))
        else:
            print("Execution time (h:min:sec): {}:{}:{}".format(tH,tM,tS))
    print("Execution time (s): {}".format(t))
    print("=====================================================")