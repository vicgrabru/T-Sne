import numpy as np

import mytsnelib.utils as ut
import time

def test_haversine():
    assert True

#=================================================================#
n_samples = 300
index_start = 0
data, labels = ut.read_csv("data/digits.csv", has_labels=True, samples=n_samples, index_start=index_start)
#=================================================================#


#======================================================#
#---Mostrar el embedding-------------------------------#
display_embed = True
#======================================================#

def probar_otra_cosa():
    print("Probando otra cosa")
    a = b = 1
    print("a: {}".format(a))
    print("b: {}".format(b))
    b = 2
    print("a: {}".format(a))
    print("b: {}".format(b))



import tests.comparacion as comp

#---Parametros ejecucion------------------------------------------#
caso_prueba = "sklearn"
print_tiempo = False


if print_tiempo:
    t0 = time.time_ns()

match caso_prueba:
    case "mio":
        comp.probar_mio(data, labels, display=display_embed)
        caso_testeo = "mio"
    case "sklearn":
        comp.probar_sklearn(data, labels, display=display_embed)
        caso_testeo = "skl"
    case "PCA":
        comp.probar_pca(data,labels, display=display_embed)
        caso_testeo = "pca"
    case "autoencoders":
        comp.probar_autoencoder(data, labels, display=display_embed)
        caso_testeo = "autoencoders"
    case _:
        probar_otra_cosa()
        caso_testeo = "otro"

if print_tiempo:
    t = (time.time_ns()-t0)*1e-9
    print("Tiempo de ejecucion {} (s): {}".format(caso_testeo, t))

