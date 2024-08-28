import numpy as np

import mytsnelib.utils as ut
import time


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

#===Mio======================================================================#
def probar_mio(data, labels, *, display=False, title=None, caso:str=None):
    from mytsnelib import functions
    model = functions.TSne(n_dimensions=n_dimensions,
                           perplexity=perplexity,
                           perplexity_tolerance=perplexity_tolerance,
                           max_iter=max_iter,
                           verbose=nivel_verbose,
                           seed=seed,
                           iters_check=iters_check)
    data_embedded = model.fit(data,classes=labels, compute_cost=calcular_coste, measure_efficiency=medir_rendimiento)
    
    if display:
        match caso:
            case "cost":
                indice = np.argmin(model.cost_history)
                embed = model.cost_history_embed[indice]
                if indice<len(model.cost_history)-1:
                    title = "Iteración con mejor coste: {}/{}".format(max(0, model.iters_check*(indice-1)), max_iter)
                else:
                    title = "Iteración con mejor coste: {}".format(max_iter)
            case _:
                title = "Mostrando embedding final"
                embed = np.copy(data_embedded)
        ut.display_embed(embed, labels, title=title)
    
    if print_cost_history:
        historia_coste = np.array(model.cost_history)
        if len(historia_coste)>0:
            print("Cost history:")
            print(historia_coste.__str__())
            for i in range(1, len(historia_coste)):
                if historia_coste[i]>historia_coste[i-1]:
                    print("--------------------------------------------------------------")
                    print("El coste en el indice {} es mayor que el previo, indice {}".format(i, i-1))
                    print("Coste con indice {}: {}".format(i-1, historia_coste[i-1]))
                    print("Coste con indice {}: {}".format(i, historia_coste[i]))
            print("--------------------------------------------------------------")
            print("Tamaño del historial de costes: {}".format(len(historia_coste)))
            print("Coste minimo, con indice {}: {}".format(np.argmin(historia_coste), np.min(historia_coste)))

#===Scikit-learn=============================================================#
def probar_sklearn(data, labels, *, display=False, title=None):
    import sklearn.manifold as mnf
    model = mnf.TSNE(n_components=n_dimensions,
                     learning_rate='auto',
                     init='random',
                     perplexity=perplexity,
                     verbose=nivel_verbose)
    data_embedded = model.fit_transform(data)

    if display:
        ut.display_embed(data_embedded, labels, title=title)

#===PCA======================================================================#
def probar_pca(data, labels, *, display=False, title=None):
    from sklearn.decomposition import PCA
    random_state = np.random.RandomState(seed)
    pca = PCA(n_components=n_dimensions, svd_solver="randomized", random_state=random_state)
    # Always output a numpy array, no matter what is configured globally
    pca.set_output(transform="default")
    data_embedded = pca.fit_transform(data).astype(np.float32, copy=False)
    # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
    # the default value for random initialization. See issue #18018.
    data_embedded = data_embedded / np.std(data_embedded[:, 0]) * 1e-4

    if display:
        ut.display_embed(data_embedded, labels, title=title)

#===Autoencoder==============================================================#
def probar_autoencoder(data, labels, *, display=False, title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from keras import layers, losses
    from keras.api.datasets import fashion_mnist
    from keras.api.models import Model
    

    # if display:
    #     ut.display_embed(data_embedded, labels, title=title)


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
