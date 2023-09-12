#from mytsnelib import functions
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A0 = np.sum(A, axis=0)
A1 = np.sum(A, axis=1)

print(A)
print(A0)
print(A1)

"""

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(random.normal(loc=0, scale=2, size=(2,3)))

plt.show()


"""



def test_haversine():
    

    assert True