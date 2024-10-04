import numpy as np
# import tests.comparacion as comp
# import tests.utils as ut
from tests import comparacion, utils
import gc

#=================================================================#
n_type = np.int16
first_column = True
skip_start = True
usar_rutas_fragmentadas = False
#-----------------------------------------------------------------
ruta_base = "tests/data/mnist_train_p{}.csv"
routes_train = []
for i in range(7):
    routes_train.append(ruta_base.format(i))
# routes_train = ["data/mnist_train_p1.csv", "data/mnist_train_p2.csv", "data/mnist_train_p3.csv", "data/mnist_train_p4.csv", "data/mnist_train_p5.csv", "data/mnist_train_p6.csv"]
if usar_rutas_fragmentadas:
    data_train, labels_train = utils.read_csv(routes_train, labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
    data_test, labels_test = utils.read_csv("tests/data/mnist_test.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
else:
    data_train, labels_train = utils.read_csv("tests/data_full/mnist_train.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
    data_test, labels_test = utils.read_csv("tests/data_full/mnist_test.csv", labels_in_first_column=first_column, num_type=n_type, skip_start_row=skip_start)
#-----------------------------------------------------------------
seed = 2
n_samples = 600
index_start = 0
rng = np.random.default_rng(seed)
usar_conjunto_entero = False
if usar_conjunto_entero:
    data_full = np.append(data_train,data_test,axis=0)
    labels_full = np.append(labels_train,labels_test,axis=0)

    random_indexes = rng.integers(index_start, len(data_full), size=n_samples)
    data_entrenamiento = data_full[random_indexes, :]
    labels_entrenamiento = labels_full[random_indexes]
    del data_full,labels_full
else:
    random_indexes = rng.integers(index_start, len(data_train), size=n_samples)
    data_entrenamiento = data_train[random_indexes, :]
    labels_entrenamiento = labels_train[random_indexes]
del random_indexes
#=================================================================#

#=================================================================#
#---Parametros ejecucion-------
caso_prueba = "mio"

parametros_train = [data_entrenamiento, labels_entrenamiento]
parametros_print = {
    "display": False,
    "print_tiempo": True,
    "trust": True,
}

#=================================================================#

match caso_prueba:
    case "mio":
        comparacion.probar_mio(*parametros_train, **parametros_print)
    case "skl":
        comparacion.probar_sklearn(*parametros_train, **parametros_print)
    case "open":
        comparacion.probar_open(*parametros_train, **parametros_print)
    case "pca":
        comparacion.probar_pca(*parametros_train, **parametros_print)
    case "autoencoders":
        comparacion.probar_autoencoder(data_entrenamiento, *parametros_train, display_amount=1000, **parametros_print)
    case "dims":
        print("data.ndim: {}".format(input.ndim))
        print("Data shape: {}".format(input.shape))
        input2 = np.ravel(input)
        print("data.flatten.shape: {}".format(input2.shape))
    case "otro":
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        a = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]], [[17, 18], [19, 20], [21, 22], [23, 24]], [[25, 26], [27, 28], [29, 30], [31, 32]]])
        b = np.array([[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6], [7, 7], [8, 8]], [[9, 9], [10, 10], [11, 11], [12,12]], [[13, 13], [14, 14], [15, 15], [16, 16]]])
        c = np.subtract.outer(a, b)

        fig, ax = plt.subplots()

        

        ani = animation.FuncAnimation()

        

        print("=================")
        print("a.shape: {}".format(a.shape))
        print("b.shape: {}".format(b.shape))
        print("c.shape: {}".format(c.shape))
        np.einsum('', a, -a)
        print("=================")
        # print("a:\n{}".format(a))
        # print("b:\n{}".format(b))
        # print("c:\n{}".format(c))

        #a -> (n, k)
        #b -> (n, k)

        # c = np.empty(len(a),len(b))
        # for i in range(len(a)):
        #     for j in range(len(b)):
        #         c[i,j] = np.subtract(a[i], b[j])

        # i1 = 1
        # j1 = 2
        # k1 = 0
        # i2 = 3
        # j2 = 1
        # k2 = 1


        # print("a[{}][{}][{}]: {}".format(i1, j1, k1, a[i1][j1][k1]))
        # print("b[{}][{}][{}]: {}".format(i2, j2, k2, b[i2][j2][k2]))
        # print("c[{}][{}][{}][{}][{}][{}]: {}".format(i1, j1, k1, i2, j2, k2, c[i1][j1][k1][i2][j2][k2]))
        # print("c[{}][{}][{}][{}][{}][{}]: {}".format(i2, j2, k2, i1, j1, k1, c[i2][j2][k2][i1][j1][k1]))
        # print("=================")

    case _:
        gc.collect()



# del caso_prueba,cost,display_embed,print_cost_evo,print_tiempo
# del data_entrenamiento,labels_entrenamiento,random_indexes,rng,index_start,n_samples,seed
# del data_full,labels_full,data_test,data_train,labels_train,labels_test
# del routes_train,usar_rutas_fragmentadas,skip_start,first_column,n_type


# gc.collect()

