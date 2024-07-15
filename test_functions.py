import numpy as np


from mytsnelib import functions
import sklearn.manifold as mnf
import mytsnelib.utils as ut
import time

#======================================================#
n_dimensions = 2
perplexity = 50
perplexity_tolerance = 1e-10
n_neighbors = 10
metric = "euclidean"
init_method = "random"
init_embed = None
early_exaggeration = 4
learning_rate = 200
max_iter = 1000
momentum_params = [250.0, 0.5, 0.8]
seed = 4
iters_check = 50
#======================================================#




def test_haversine():
    

    assert True


def probar_mio(data, labels, *, verbose=1, display=None, title=None, compute_extra=False, print_trust=False):
    model = functions.TSne(n_dimensions=n_dimensions,
                           perplexity=perplexity,
                           perplexity_tolerance=perplexity_tolerance,
                           n_neighbors=n_neighbors,
                           max_iter=max_iter,
                           verbose=verbose,
                           seed=seed,
                           iters_check=iters_check)
    data_embedded = model.fit(data,classes=labels, compute_cost_trust=compute_extra)
    if display is not None:
        if display=="last":
            model.display_embed(t=-1, title=title)
        elif display=="cost":
            model.display_embed(display_best_iter_cost=True, title=title)
        elif display=="trust":
            model.display_embed(display_best_iter_trust=True, title=title)
    if print_trust:
        trust =  functions.trustworthiness_safe(data, data_embedded, n_neighbors)
        #trust =  functions.trustworthiness_fast(data, data_embedded, n_neighbors)
        print("===============================================================================")
        print("trust con el mio: {}".format(trust))
        print("===============================================================================")

def probar_sklearn(data, labels, *, verbose=1, display=False, title=None, print_trust=False):
    data_embedded = mnf.TSNE(n_components=n_dimensions,
                         learning_rate='auto',
                         init='random',
                         perplexity=perplexity,
                         verbose=verbose).fit_transform(data)
    if display:
        ut.display_embed(data_embedded, labels, title=title)
    if print_trust:
        trust = mnf.trustworthiness(data, data_embedded, n_neighbors=n_neighbors)
        print("===============================================================================")
        print("trust con el de sklearn: {}".format(trust))
        print("===============================================================================")
        



#=======================================================================================================#
#===========================TODO: terminar de implementar el metodo de prueba===========================#
#=======================================================================================================#
def probar_bht(data, labels, *, verbose=1, display=False, title=None):
    import bhtsne
    embedding_bht = bhtsne.tsne(data, initial_dims=data.shape[1])
    

def probar_open(data, labels, *, verbose=1, display=False, title=None):
    import openTSNE
    embedding_open = openTSNE.TSNE(n_iter=max_iter, n_components=n_dimensions, perplexity=perplexity).fit(data)
#=======================================================================================================#
#=======================================================================================================#
#=======================================================================================================#



def probar_otra_cosa():
    print("Probando otra cosa")
    a=np.array([[1,2],[1,2],[1,2],[1,2]])

    a0 = np.expand_dims(a,0)
    a1 = np.expand_dims(a,1)
    a2 = np.expand_dims(a,2)


    # y = (n_samples,n_dims)
    # expand_dims -> n_dims +=1

    print("----------------------")
    print("a.shape: {}".format(a.shape))
    print("a:\n {}".format(a))
    print("----------------------")
    print("a0.shape: {}".format(a0.shape))
    print("a0:\n {}".format(a0))
    print("----------------------")
    print("a1.shape: {}".format(a1.shape))
    print("a1:\n {}".format(a1))
    print("----------------------")
    print("a2.shape: {}".format(a2.shape))
    print("a2:\n {}".format(a2))
    print("----------------------")


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#POSIBLE SOLUCION PARA ELIMINAR EL BUCLE ANIDADO DE SACAR LOS INDICES DE LOS VECINOS MAS CERCANOS
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def probar_otra_cosa_2():
    
    c = (2,3)
    indices_c = np.indices(c)
    
    x = np.arange(20).reshape(5, 4)
    row, col = np.indices((2, 3))

    print("===============================")
    print("x = np.arange(20).reshape(5, 4)")
    print("-------------------------------")
    print("x: \n{}".format(x))
    #  x:
    # [[ 0  1  2  3]
    #  [ 4  5  6  7]
    #  [ 8  9 10 11]
    #  [12 13 14 15]
    #  [16 17 18 19]]
    print("-------------------------------")
    print("row, col = np.indices((2, 3))")
    print("row: \n{}".format(row))
    print("-------------------------------")
    print("col: \n{}".format(col))
    print("-------------------------------")
    print("x[row, col]: \n{}".format(x[row, col]))
    # x[row, col]:
    # [[0 1 2]
    #  [4 5 6]]
    print("np.indices((2,3)): \n{}".format(indices_c))
    print("===============================")


def comparacion_tiempos(data, labels, *, mio=False, sklearn=False, bht=False, open=False):
    print("=======================================")
    if mio:
        t0 = time.time()
        probar_mio(data, labels, verbose=0)
        t1 = time.time()
        t_diff = t1-t0
        print("mytsnelib:")
        print("Tiempo de ejecucion: {}".format(time.strftime("%H:%M:%S", time.gmtime(t_diff))))
        print("=======================================")
    if sklearn:
        t0 = time.time()
        probar_sklearn(data, labels, verbose=0)
        t1 = time.time()
        t_diff = t1-t0
        print("sklearn:")
        print("Tiempo de ejecucion: {}".format(time.strftime("%H:%M:%S", time.gmtime(t_diff))))
        print("=======================================")
    if bht:
        t0 = time.time()
        probar_bht(data, labels, verbose=0)
        t1 = time.time()
        t_diff = t1-t0
        print("bht-sne:")
        print("Tiempo de ejecucion: {}".format(time.strftime("%H:%M:%S", time.gmtime(t_diff))))
        print("=======================================")
    if open:
        t0 = time.time()
        probar_open(data, labels, verbose=0)
        t1 = time.time()
        t_diff = t1-t0
        print("openTSNE:")
        print("Tiempo de ejecucion: {}".format(time.strftime("%H:%M:%S", time.gmtime(t_diff))))
        print("=======================================")

def comparacion_resultados(data, labels, *, mio=False, sklearn=False, bht=False, open=False, caso_mio="last"):
    if mio:
        probar_mio(data, labels, verbose=0, display=caso_mio, title="Mytsnelib")
    if sklearn:
        probar_sklearn(data, labels, verbose=0, display=True, title="Sklearn")
    if bht:
        probar_bht(data, labels, verbose=0, display=True, title="bht-sne")
    if open:
        probar_open(data, labels, verbose=0, display=True, title="OpenTSNE")

def comparacion_trust(data, labels, *, mio=False, sklearn=False):
    if mio:
        probar_mio(data, labels, verbose=0, compute_extra=True, print_trust=True)
    if sklearn:
        probar_sklearn(data, labels, verbose=0, print_trust=True)
    


def prueba_individual(data, labels, *, caso, display=None, compute_extra=True):
    if caso=="mio":
        probar_mio(data, labels, display=display, compute_extra=compute_extra)
    elif caso=="sklearn":
        probar_sklearn(data, labels)
    elif caso=="bht":
        probar_bht(data, labels)
    elif caso=="open":
        probar_open(data, labels)
    else:
        print("Caso predeterminado")


#=================================================================#
include_n_samples = 300
index_start = 0

read_csv = ut.read_csv("data/digits.csv", has_labels=True)
data_full = read_csv[0].astype(np.int32)
labels_full = read_csv[1]
data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]
#=================================================================#


#comparacion_resultados(data, labels)

#comparacion_trust(data, labels, mio=True, sklearn=True)

prueba_individual(data, labels, caso="mio", compute_extra=False)

#probar_otra_cosa()

#probar_otra_cosa_2()


