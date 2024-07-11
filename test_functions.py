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
momentum_params = [1.0, 0.5, 0.8]
n_dimensions = 2
n_iterations = 1000
neighbors = 10
perplexity_tolerance = 1e-10
perplexity = 3
verbose=1
seed = 3
#======================================================#


read_csv = ut.read_csv("data/digits.csv", has_labels=True)
data_full = read_csv[0].astype(np.int32)
labels_full = read_csv[1]
data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]


#1: mio
#2: sklearn
#3: pagina
#4: pagina en functions.py
#5: pagina con los datos de prueba mios
#6: pagina en functions.py con los datos de prueba mios
#default: otra cosa

probando = 1
if probando==1:
    print("Probando T-Sne mio")
    model = functions.TSne(n_dimensions=n_dimensions,
                           perplexity_tolerance=perplexity_tolerance,
                           max_iter=n_iterations,
                           n_neighbors=neighbors,
                           perplexity=perplexity,
                           verbose=verbose,
                           seed=seed)

    model.fit(data,classes=labels)

    model.display_embed(t=-1)
    # model.display_embed(display_best_iter_cost=True)
    # model.display_embed(display_best_iter_trust=True)

    
    
    # trustworthinessCost = functions.trustworthiness(data, model.Y[model.best_iter_cost], model.n_neighbors)
    # print("======================================================")
    # print("===========Cost=======================================")
    # print("best_iter_cost:{}".format(model.best_iter_cost))
    # print("best_cost:{}".format(model.best_cost))
    # print("trustworthiness of the best cost embedding: {}".format(trustworthinessCost))
    # print("======================================================")


    # print("======================================================")
    # print("===========Trust======================================")
    # trustworthinessTrust = functions.trustworthiness(data, model.Y[model.best_iter_trust], model.n_neighbors)
    # print("best_iter_trust:{}".format(model.best_iter_trust))
    # print("best_trust:{}".format(model.best_trust))
    # print("trustworthiness of the best trust embedding: {}".format(trustworthinessTrust))
    # print("======================================================")
    # print("======================================================")

elif probando==2:
    print("Probando T-Sne de sklearn")
    from sklearn.manifold import TSNE
    data_embedded = TSNE(n_components=n_dimensions,
                         learning_rate='auto',
                         init='random',
                         perplexity=perplexity,
                         verbose=verbose).fit_transform(data)
    
    ut.display_embed(data_embedded, labels)

elif probando==3:
    print("Probando T-Sne de la pagina")
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    res = mtdpg.tsne(X, T=1000, l=200, perp=40)
    plt.scatter(res[:, 0], res[:, 1], s=20, c=y)
    plt.show()

elif probando==4:
    print("Probando T-Sne de la pagina en functions.py")
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    res = functions.tsne_2(X, T=1000, l=200, perp=40)
    plt.scatter(res[:, 0], res[:, 1], s=20, c=y)
    plt.show()

elif probando==5:
    print("Probando T-Sne de la pagina con los datos de prueba mios")
    res = mtdpg.tsne(data, T=1000, l=200, perp=40)
    plt.scatter(res[:, 0], res[:, 1], s=20)
    plt.show()

elif probando==6:
    print("Probando T-Sne de la pagina en functions.py con los datos de prueba mios")
    res = functions.tsne_2(data, T=1000, l=200, perp=40)
    plt.scatter(res[:, 0], res[:, 1], s=20)
    plt.show()


else:
    print("Probando otra cosa")
    a=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])

    a0 = np.expand_dims(a,0)
    a1 = np.expand_dims(a,1)
    a2 = np.expand_dims(a,2)


    # y = (n_samples,n_dims)
    # expand_dims -> n_dims +=1

    print("----------------------")
    print("a.ndim: {}".format(a.ndim))
    print("a.shape: {}".format(a.shape))
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


def test_haversine():
    

    assert True