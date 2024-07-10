import numpy as np


from mytsnelib import functions, similarities
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
perplexity_tolerance = 0.
perplexity = 3
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
    data_embedded = TSNE(n_components=n_dimensions, learning_rate='auto', init='random', perplexity=perplexity, verbose=1).fit_transform(data)
    ut.display_embed(data_embedded, labels)
    

elif probando==2:
    print("Probando T-Sne mio")
    model = functions.TSne(n_dimensions=n_dimensions,perplexity_tolerance=perplexity_tolerance,max_iter=n_iterations,n_neighbors=neighbors,perplexity=perplexity)

    model.fit(data,classes=labels)

    min_cost = min(model.cost_history)
    best_iter = np.where(model.cost_history==min_cost)[0][0]
    best_embed = model.embedding_history[best_iter]

    print("best_iter:{}".format(best_iter))
    print("min_cost:{}".format(min_cost))
    print("best_cost:{}".format(model.best_cost))

    trustworthiness = functions.trustworthiness(data, best_embed, model.n_neighbors)
    
    print("trustworthiness: {}".format(trustworthiness))

    
    model.display_embed(-5)

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


def test_haversine():
    

    assert True