import numpy as np

def euclidean_distance(X, Y=None, X_dot_products=None, Y_dot_products=None, return_squared=False):
    X, Y = check_arrays_compatible(X,Y)
    if X_dot_products is not None:
        if X_dot_products.shape[1] != X.shape[0]:
            raise ValueError("X_dot_products must be of shape (X.rows, 1)")
    else:
        X_dot_products = (X*X).sum(axis=1)
    if Y_dot_products is not None:
        if Y_dot_products.shape[1] != Y.shape[0]:
            raise ValueError("Y_dot_products must be of shape (Y.rows, 1)")
    else:
        Y_dot_products = (Y * Y).sum(axis=1)
    return __euclidian_distance(X,Y,X_dot_products,Y_dot_products,return_squared)

def __euclidian_distance(X, Y, X_dot_products, Y_dot_products, return_squared):
    XX = X_dot_products.reshape(-1,1)
    YY = Y_dot_products.reshape(1,-1)


    dist = -2 * np.dot(X,Y)
    dist += XX
    dist += YY

    return dist if not return_squared else np.sqrt(dist, out=dist)


def check_arrays_compatible(X,Y=None):
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if Y==None:
        Y = X
    elif not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Arrays must contain the same number of columns")
    return X,Y