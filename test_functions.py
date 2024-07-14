import numpy as np



import mytsnelib.utils as ut
import time

#======================================================#
n_dimensions = 2
perplexity = 40
perplexity_tolerance = 1e-10
n_neighbors = 10
metric = "euclidean"
init_method = "random"
init_embed = None
early_exaggeration = 4
learning_rate = 200
max_iter = 1000
momentum_params = [1.0, 0.5, 0.8]
seed = 3
iters_check = 1
#======================================================#




def test_haversine():
    

    assert True


def probar_mio(data, labels, *, verbose=1, display=None, title=None):
    from mytsnelib import functions
    model = functions.TSne(n_dimensions=n_dimensions,
                           perplexity=perplexity,
                           perplexity_tolerance=perplexity_tolerance,
                           n_neighbors=n_neighbors,
                           max_iter=max_iter,
                           verbose=verbose,
                           seed=seed,
                           iters_check=iters_check)
    model.fit(data,classes=labels)
    if display is not None:
        if display=="last":
            model.display_embed(t=-1, title=title)
        elif display=="cost":
            model.display_embed(display_best_iter_cost=True, title=title)
        elif display=="trust":
            model.display_embed(display_best_iter_trust=True, title=title)

def probar_sklearn(data, labels, *, verbose=1, display=False, title=None):
    from sklearn.manifold import TSNE
    data_embedded = TSNE(n_components=n_dimensions,
                         learning_rate='auto',
                         init='random',
                         perplexity=perplexity,
                         verbose=verbose).fit_transform(data)
    if display:
        ut.display_embed(data_embedded, labels, title=title)



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

def prueba_individual(data, labels, *, caso, display=None):
    if caso=="mio":
        probar_mio(data, labels, display=display)
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

prueba_individual(data, labels, caso="mio", display="cost")




#probar_otra_cosa()


