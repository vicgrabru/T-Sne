import numpy as np
import mytsnelib.utils as ut
import time
from os.path import join

def test_haversine():
    assert True

#=================================================================#
n_samples = 300
index_start = 0

data_train_1, labels_train_1 = ut.read_csv("data/mnist_train_p1.csv", has_labels=True, labels_in_first_column=True)
data_train_2, labels_train_2 = ut.read_csv("data/mnist_train_p2.csv", has_labels=True, labels_in_first_column=True)
data_train_3, labels_train_3 = ut.read_csv("data/mnist_train_p3.csv", has_labels=True, labels_in_first_column=True)
data_train_4, labels_train_4 = ut.read_csv("data/mnist_train_p4.csv", has_labels=True, labels_in_first_column=True)
data_train_5, labels_train_5 = ut.read_csv("data/mnist_train_p5.csv", has_labels=True, labels_in_first_column=True)
data_train_6, labels_train_6 = ut.read_csv("data/mnist_train_p6.csv", has_labels=True, labels_in_first_column=True)
data_test, labels_test = ut.read_csv("data/mnist_test.csv", has_labels=True, labels_in_first_column=True)

data_full = np.append(data_train_1, data_train_2, data_train_3, data_train_4, data_train_5, data_train_6, data_test, axis=0)
labels_full = np.append(labels_train_1, labels_train_2, labels_train_3, labels_train_4, labels_train_5, labels_train_6, labels_test, axis=0)

data = data_full[index_start:index_start+n_samples]
labels = labels_full[index_start:index_start+n_samples]
#=================================================================#



prueba_algoritmo = True

def test_data_dims(input:np.ndarray):
    print("data.ndim: {}".format(input.ndim))
    print("Data shape: {}".format(input.shape))
    input2 = np.ravel(input)
    print("data.flatten.shape: {}".format(input2.shape))


def probar_otra_cosa():
    super.prueba_algoritmo=False
    print("Probando otra cosa")
    a = b = 1
    print("a: {}".format(a))
    print("b: {}".format(b))
    b = 2
    print("a: {}".format(a))
    print("b: {}".format(b))

    c = 5
    d = 6
    e = -8

    print("c: {}".format(c))
    print("d: {}".format(d))
    print("e: {}".format(e))
    print("{} % {}: {}".format(e,c,e%c))


#======================================================#
#---Mostrar el embedding-------------------------------#
display_embed = True
#======================================================#

import tests.comparacion as comp

#---Parametros ejecucion------------------------------------------#
caso_prueba = "dims"
print_tiempo = False



if print_tiempo:
    t0 = time.time_ns()

match caso_prueba:
    case "mio":
        comp.probar_mio(data, labels, display=display_embed)
        # comp.probar_mio(data_train, labels_train, display=display_embed)
    case "skl":
        comp.probar_sklearn(data, labels, display=display_embed)
    case "PCA":
        comp.probar_pca(data,labels, display=display_embed)
    case "autoencoders":
        comp.probar_autoencoder(data, labels, display=display_embed)
    case "dims":
        test_data_dims(data)
    case _:
        probar_otra_cosa()

if print_tiempo and prueba_algoritmo:
    t = (time.time_ns()-t0)*1e-9
    print("Tiempo de ejecucion {} (s): {}".format(caso_prueba, t))

