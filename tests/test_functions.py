








#======================================================#
include_n_samples = 300
index_start = 0
#======================================================#


#======================================================#
n_dimensions = 2
n_iterations = 1000
neighbors = None
perplexity_tolerance = 0.
perplexity = 50
#======================================================#

import numpy as np
from mytsnelib import functions
import mytsnelib.utils as ut
read_csv = ut.read_csv("data/digits.csv", has_labels=True)
data_full = read_csv[0].astype(np.int64)
labels_full = read_csv[1]
data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]
model = functions.TSne(n_dimensions=n_dimensions,
                       perplexity_tolerance=perplexity_tolerance,
                       max_iter=n_iterations,
                       n_neighbors=neighbors,
                       perplexity=perplexity)


model.fit(data, classes=labels)





model.display_embed()




def test_haversine():
    

    assert True