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
#---Parametros para entrenar el modelo-----------------#
n_dimensions = 2
perplexity = 30.
perplexity_tolerance = 1e-10
metric = "euclidean"
init_method = "random"
init_embed = None
early_exaggeration = 12.
learning_rate = 200.
max_iter = 5000
momentum_params = [250., 0.5, 0.8]
seed = 4
iters_check = 50
#---Parametros para calculos extra---------------------#
calcular_coste=False
#---Cosas que mostrar por consola----------------------# 
medir_rendimiento=False
print_cost_history=False
nivel_verbose=0
#---Mostrar el embedding-------------------------------#
display_embed = True
mostrar_resultado = "last" # None para no mostrar, "last" para mostrar la ultima, "cost" para mostrar la que obtiene mejor coste
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
        comp.probar_mio(data, labels)
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

