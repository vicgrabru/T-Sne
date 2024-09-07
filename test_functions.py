import numpy as np
import mytsnelib.utils as ut
import time
import tests.comparacion as comp


# from os.path import join

def test_haversine():
    assert True

#=================================================================#
n_type = np.int16
first_column = True
skip_start = True

# 
# routes_data = ["data/mnist_train_p1.csv", "data/mnist_train_p2.csv", "data/mnist_train_p3.csv", "data/mnist_train_p4.csv", "data/mnist_train_p5.csv", "data/mnist_train_p6.csv", "data/mnist_test.csv"]
# routes_train = ["data/mnist_train_p1.csv", "data/mnist_train_p2.csv", "data/mnist_train_p3.csv", "data/mnist_train_p4.csv", "data/mnist_train_p5.csv", "data/mnist_train_p6.csv"]
# 
# data_full, labels_full = ut.read_csv(routes_data, labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
# data_full, labels_full = ut.read_csv(routes_train, labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
# data_test, labels_test = ut.read_csv("data/mnist_test.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
# 
# 
# data = data_full[random_indexes]
# labels = labels_full[random_indexes]

data, labels = ut.read_csv("data_full/mnist_test.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
data_test, labels_test = ut.read_csv("data_full/mnist_test.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)

seed = 2
n_samples = 600
index_start = 0
rng = np.random.default_rng(seed)
random_indexes = rng.integers(index_start, len(data), size=n_samples)

data_prueba = data[random_indexes, :]
labels_prueba = labels[random_indexes]
#=================================================================#



prueba_algoritmo = True

def test_data_dims(input:np.ndarray):
    print("data.ndim: {}".format(input.ndim))
    print("Data shape: {}".format(input.shape))
    input2 = np.ravel(input)
    print("data.flatten.shape: {}".format(input2.shape))


def probar_otra_cosa():
    
    
    print("Probando otra cosa")
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])


#---Parametros ejecucion------------------------------------------#
print_tiempo = False
#---------------------------
caso_prueba = "mio"
display_embed = True
compute_cost = True
display_cost = True
print_cost_evo = False

if print_tiempo:
    t0 = time.time_ns()

match caso_prueba:
    case "mio":
        comp.probar_mio(data_prueba, labels_prueba, display=display_embed, calcular_coste=compute_cost, display_best_cost=display_cost, print_cost_history=print_cost_evo)
        # comp.probar_mio(data_full, labels_full, display=display_embed)
    case "skl":
        comp.probar_sklearn(data_prueba, labels_prueba, display=display_embed)
        # comp.probar_sklearn(data_full, labels_full, display=display_embed)
    case "pca":
        comp.probar_pca(data,labels, display=display_embed)
        # comp.probar_pca(data_full,labels_full, display=display_embed)
    case "autoencoders":
        # comp.probar_autoencoder(data, test_data=data_test, test_labels=labels_test, display=display_embed)
        comp.probar_autoencoder(data, test_data=data_test, test_labels=labels_test, display=display_embed, display_amount=1000)
        # comp.probar_autoencoder(data_full, labels_full, test_data=data_test, test_labels=labels_test, display=display_embed)
    case "dims":
        test_data_dims(data)
    case _:
        probar_otra_cosa()

if print_tiempo and prueba_algoritmo:
    t = (time.time_ns()-t0)*1e-9
    print("Tiempo de ejecucion {} (s): {}".format(caso_prueba, t))

