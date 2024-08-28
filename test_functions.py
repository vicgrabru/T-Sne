import numpy as np
import mytsnelib.utils as ut
import time
from os.path  import join

def test_haversine():
    assert True

#=================================================================#
n_samples = 300
index_start = 0
data, labels = ut.read_csv("data/digits.csv", has_labels=True, samples=n_samples, index_start=index_start)
input_path = 'data\MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = ut.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(data_train, labels_train), (data_test, labels_test) = mnist_dataloader.load_data()
#=================================================================#



prueba_algoritmo = True



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
caso_prueba = "mio"
print_tiempo = False



if print_tiempo:
    t0 = time.time_ns()

match caso_prueba:
    case "mio":
        comp.probar_mio(data, labels, display=display_embed)
    case "skl":
        comp.probar_sklearn(data, labels, display=display_embed)
    case "PCA":
        comp.probar_pca(data,labels, display=display_embed)
    case "autoencoders":
        comp.probar_autoencoder(data, labels, display=display_embed)
    case _:
        probar_otra_cosa()

if print_tiempo and prueba_algoritmo:
    t = (time.time_ns()-t0)*1e-9
    print("Tiempo de ejecucion {} (s): {}".format(caso_prueba, t))

