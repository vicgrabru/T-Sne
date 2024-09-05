import numpy as np
import mytsnelib.utils as ut
import time
# from os.path import join

def test_haversine():
    assert True

#=================================================================#
n_samples = 300
index_start = 0

routes_data = ["data/mnist_train_p1.csv", "data/mnist_train_p2.csv", "data/mnist_train_p3.csv", "data/mnist_train_p4.csv", "data/mnist_train_p5.csv", "data/mnist_train_p6.csv", "data/mnist_test.csv"]
routes_train = ["data/mnist_train_p1.csv", "data/mnist_train_p2.csv", "data/mnist_train_p3.csv", "data/mnist_train_p4.csv", "data/mnist_train_p5.csv", "data/mnist_train_p6.csv"]

# data_full, labels_full = ut.read_csv(routes_data, labels_in_first_column=True, num_type=np.int8, skip_start_row=True)
data_full, labels_full = ut.read_csv(routes_train, labels_in_first_column=True, num_type=np.int8, skip_start_row=True)
data_test, labels_test = ut.read_csv("data/mnist_test.csv", labels_in_first_column=True, num_type=np.int8, skip_start_row=True)

# data_full, labels_full = ut.read_csv("data/mnist_train_p1.csv", labels_in_first_column=True, num_type=np.int8, skip_start_row=True)

random_indexes = np.random.randint(0, len(data_full), size=n_samples)

# data = data_full[index_start:index_start+n_samples]
# labels = labels_full[index_start:index_start+n_samples]

data = data_full[random_indexes]
labels = labels_full[random_indexes]

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
display_embed = False
#======================================================#

import tests.comparacion as comp

#---Parametros ejecucion------------------------------------------#
caso_prueba = "autoencoders"
print_tiempo = True



if print_tiempo:
    t0 = time.time_ns()

match caso_prueba:
    case "mio":
        comp.probar_mio(data, labels, display=display_embed)
        # comp.probar_mio(data_full, labels_full, display=display_embed)
    case "skl":
        # comp.probar_sklearn(data, labels, display=display_embed)
        comp.probar_sklearn(data_full, labels_full, display=display_embed)
    case "PCA":
        # comp.probar_pca(data,labels, display=display_embed)
        comp.probar_pca(data_full,labels_full, display=display_embed)
    case "autoencoders":
        #comp.probar_autoencoder(data, labels, test_labels=labels_test, display=display_embed, display=display_embed)
        comp.probar_autoencoder(data_full, labels_full, test_data=data_test, test_labels=labels_test, display=display_embed)
    case "dims":
        test_data_dims(data)
    case _:
        probar_otra_cosa()

if print_tiempo and prueba_algoritmo:
    t = (time.time_ns()-t0)*1e-9
    print("Tiempo de ejecucion {} (s): {}".format(caso_prueba, t))

