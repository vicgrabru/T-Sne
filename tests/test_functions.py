



import numpy as np
from mytsnelib import functions
import mytsnelib.utils as ut

include_n_samples = 300
index_start = 0

read_csv = ut.read_csv("data/digits.csv", has_labels=True)

data_full = read_csv[0].astype(np.int64)
labels_full = read_csv[1]

data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]



model = functions.TSne(max_iter=30)

model.fit(data, classes=labels)





model.display_embed()

"""

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(random.normal(loc=0, scale=2, size=(2,3)))

plt.show()


"""



def test_haversine():
    

    assert True