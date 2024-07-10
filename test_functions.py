import numpy as np


from mytsnelib import functions
import mytsnelib.utils as ut
import matplotlib.pyplot as plt
import mytsnelib.metodos_pagina as mtdpg

#======================================================#
include_n_samples = 300
index_start = 0
#======================================================#
#======================================================#
n_dimensions = 2
n_iterations = 1000
neighbors = 10
perplexity_tolerance = 1e-10
perplexity = 3
verbose=1
#======================================================#


read_csv = ut.read_csv("data/digits.csv", has_labels=True)
data_full = read_csv[0].astype(np.int32)
labels_full = read_csv[1]
data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]


#1: sklearn
#2: mio
#3: pagina
#default: otra cosa
probando = 2

if probando==1:
    print("Probando T-Sne de sklearn")
    from sklearn.manifold import TSNE
    data_embedded = TSNE(n_components=n_dimensions,
                         learning_rate='auto',
                         init='random',
                         perplexity=perplexity,
                         verbose=verbose).fit_transform(data)
    
    ut.display_embed(data_embedded, labels)
    

elif probando==2:
    print("Probando T-Sne mio")
    model = functions.TSne(n_dimensions=n_dimensions,
                           perplexity_tolerance=perplexity_tolerance,
                           max_iter=n_iterations,
                           n_neighbors=neighbors,
                           perplexity=perplexity,
                           verbose=verbose)

    model.fit(data,classes=labels)

    model.display_embed(t=-1)
    model.display_embed(display_best_iter_cost=True)
    model.display_embed(display_best_iter_trust=True)

    
    
    trustworthinessCost = functions.trustworthiness(data, model.Y[model.best_iter_cost], model.n_neighbors)
    print("======================================================")
    print("===========Cost=======================================")
    print("best_iter_cost:{}".format(model.best_iter_cost))
    print("best_cost:{}".format(model.best_cost))
    print("trustworthiness of the best cost embedding: {}".format(trustworthinessCost))
    print("======================================================")


    print("======================================================")
    print("===========Trust======================================")
    trustworthinessTrust = functions.trustworthiness(data, model.Y[model.best_iter_trust], model.n_neighbors)
    print("best_iter_trust:{}".format(model.best_iter_trust))
    print("best_trust:{}".format(model.best_trust))
    print("trustworthiness of the best trust embedding: {}".format(trustworthinessTrust))
    print("======================================================")
    print("======================================================")
    
    
    

elif probando==3:
    print("Probando T-Sne de la pagina")
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    res = mtdpg.tsne(data, T=1000, l=200, perp=40)
    plt.scatter(res[:, 0], res[:, 1], s=20)
    plt.show()

else:
    print("Probando otra cosa")
    a=np.array([[1,2,3],[1,2,3],[1,2,3]])

    a0 = np.repeat(a, 3, axis=0)
    a1 = np.repeat(a, 3, axis=1)

    b1 = np.array([[1,2],[3,4]])
    b2 = np.array([1,1])

    b3 = b1-b2
    b4 = b2-b1

    b5 = b1 - 1

    print("\n b1:")
    print(b1)

    print("\n b5:")
    print(b5)


    print("\n clase de False:")
    print(False.__class__)


def test_haversine():
    

    assert True