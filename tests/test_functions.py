


import numpy as np
from mytsnelib import functions, similarities
import mytsnelib.utils as ut

#======================================================#
include_n_samples = 300
index_start = 0
#======================================================#
#======================================================#
n_dimensions = 2
n_iterations = 1000
neighbors = 10
perplexity_tolerance = 0.
perplexity = 50
#======================================================#


read_csv = ut.read_csv("data/digits.csv", has_labels=True)
data_full = read_csv[0].astype(np.int32)
labels_full = read_csv[1]
data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]


model = functions.TSne(n_dimensions=n_dimensions,perplexity_tolerance=perplexity_tolerance,max_iter=n_iterations,n_neighbors=neighbors,perplexity=perplexity)



model.fit(data,classes=labels)




min_cost = min(model.cost_history)
best_iter = np.where(model.cost_history==min_cost)[0][0]
best_embed = model.embedding_history[best_iter]

print(best_iter)
print(min_cost)

trustworthiness = functions.trustworthiness(data, best_embed, model.n_neighbors)
print(trustworthiness)


a=np.array([[1,2,3],[1,2,3],[1,2,3]])

a0 = np.sum(a,axis=0)
a1 = np.sum(a,axis=1)

print(a)
print(a0)
print(a1)

print(len(a))

"""
c_i =
[a[0,i]*b[0,i]   a[0,i]*b[1,i]      ...     a[0,i]*b[n,i]]
[a[1,i]*b[0,i]   a[1,i]*b[1,i]      ...     a[1,i]*b[n,i]]
[       ...             ...         ...         ...      ]
[a[n,i]*b[0,i]   a[n,i]*b[1,i]      ...     a[n,i]*b[n,i]]
=

[a[0,i]
a[1,i]
...
a[n,i]] = a_columna_i = a[:,i-1:i]
b_columna_i(como fila) = b[:,i-1:i].T

for i in range(a.shape[1]):
    a_columna_i = a[:,i]
    b_columna_i_como_fila = b[:,i].T
    a_columna_i * b_columna_i_como_fila


a[i].T*b[i]

sum_i(a[i].T*b[i])


>>> a = [[1, 0], [0, 1]]
>>> b = [[4, 1], [2, 2]]
>>> np.dot(a, b)

[1  0]  *   [4  1]
[0  1]      [2  2]


[4, 1]
[2, 2]


dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])



"""


model.display_embed()


def test_haversine():
    

    assert True