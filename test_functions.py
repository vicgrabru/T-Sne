import numpy as np


from mytsnelib import functions
import sklearn.manifold as mnf
import mytsnelib.utils as ut
import time

def test_haversine():
    assert True

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

def probar_mio(data, labels, *, verbose=1, display=None, title=None, compute_cost=False, compute_trust=False, medir_tiempo=False):
    model = functions.TSne(n_dimensions=n_dimensions,
                           perplexity=perplexity,
                           perplexity_tolerance=perplexity_tolerance,
                           n_neighbors=n_neighbors,
                           max_iter=max_iter,
                           verbose=verbose,
                           seed=seed,
                           iters_check=iters_check)
    data_embedded = model.fit(data,classes=labels, compute_cost=compute_cost, compute_trust=compute_trust, measure_efficiency=medir_tiempo)
    if display is not None:
        if display=="last":
            model.display_embed(t=-1, title=title)
        elif display=="cost":
            model.display_embed(display_best_iter_cost=True, title=title)
        elif display=="trust":
            model.display_embed(display_best_iter_trust=True, title=title)

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


#=================================================================#
include_n_samples = 300
index_start = 0

read_csv = ut.read_csv("data/digits.csv", has_labels=True)
data_full = read_csv[0].astype(np.int32)
labels_full = read_csv[1]
data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]
#=================================================================#



t0 = time.time_ns()
probar_mio(data, labels, verbose=0, compute_cost=False, compute_trust=True)
t_diff_1 = (time.time_ns()-t0)*1e-9
print("Tiempo de ejecucion mio (s): {}".format(t_diff_1))

# t2 = time.time_ns()
# probar_sklearn(data, labels, verbose=0)
# t_diff_2 = (time.time_ns()-t2)*1e-9
# print("Tiempo de ejecucion skl (s): {}".format(t_diff_2))



