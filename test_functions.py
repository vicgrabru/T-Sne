import numpy as np
import mytsnelib.utils as ut
import tests.comparacion as comp

def test_haversine():
    assert True

#=================================================================#
n_type = np.int16
first_column = True
skip_start = True
usar_rutas_fragmentadas = False
routes_train = ["data/mnist_train_p1.csv", "data/mnist_train_p2.csv", "data/mnist_train_p3.csv", "data/mnist_train_p4.csv", "data/mnist_train_p5.csv", "data/mnist_train_p6.csv"]
if usar_rutas_fragmentadas:
    data_train, labels_train = ut.read_csv(routes_train, labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
    data_test, labels_test = ut.read_csv("data/mnist_test.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
else:
    data_train, labels_train = ut.read_csv("data_full/mnist_train.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
    data_test, labels_test = ut.read_csv("data_full/mnist_test.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)

data_full = np.append(data_train,data_test,axis=0)
labels_full = np.append(labels_train,labels_test,axis=0)




seed = 2
n_samples = 600
index_start = 0
rng = np.random.default_rng(seed)
random_indexes = rng.integers(index_start, len(data_train), size=n_samples)

data_entrenamiento = data_train[random_indexes, :]
labels_entrenamiento = labels_train[random_indexes]
#=================================================================#

def test_data_dims(input:np.ndarray):
    print("data.ndim: {}".format(input.ndim))
    print("Data shape: {}".format(input.shape))
    input2 = np.ravel(input)
    print("data.flatten.shape: {}".format(input2.shape))

def probar_otra_cosa():
    
    
    print("Probando otra cosa")
    a = np.array([
                    [[1,2,3],[4,5,6],[7,8,9]],
                    [[10,11,12],[13,14,15],[16,17,18]],
                    [[19,20,21],[22,23,24],[25,26,27]]
                  ])
    a0 = np.sum(a, axis=0)
    a1 = np.sum(a, axis=1)
    a2 = np.sum(a, axis=2)

    b = np.array([[1,2,3],[4,5,6],[7,8,9]])
    c = np.array([[2,3,4],[1,0,0],[5,5,5]])
    b2 = np.expand_dims(b, 2)

#=================================================================#
#---Parametros ejecucion-------
caso_prueba = "mio"
#---Calculos extra-------------
compute_cost = True
#---Parametros de display------
display_embed = True
display_cost = True
#---Salida por consola---------
print_cost_evo = False
print_tiempo = True

#=================================================================#

match caso_prueba:
    case "mio":
        comp.probar_mio(data_entrenamiento, labels_entrenamiento, display=display_embed, calcular_coste=compute_cost, display_best_cost=display_cost, print_cost_history=print_cost_evo, print_tiempo=print_tiempo)
    case "skl":
        comp.probar_sklearn(data_entrenamiento, labels_entrenamiento, display=display_embed, print_tiempo=print_tiempo)
    case "pca":
        comp.probar_pca(data_train,labels_train, display=display_embed, print_tiempo=print_tiempo)
    case "autoencoders":
        comp.probar_autoencoder(data_train, test_data=data_test, test_labels=labels_test, display=display_embed, display_amount=1000, print_tiempo=print_tiempo)
    case "dims":
        test_data_dims(data_entrenamiento)
    case _:
        probar_otra_cosa()



del caso_prueba,compute_cost,display_embed,display_cost,print_cost_evo,print_tiempo
del data_entrenamiento,labels_entrenamiento,random_indexes,rng,index_start,n_samples,seed
del data_full,labels_full,data_test,data_train,labels_train,labels_test
del routes_train,usar_rutas_fragmentadas,skip_start,first_column,n_type

import gc
gc.collect()

