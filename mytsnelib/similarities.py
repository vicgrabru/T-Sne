import numpy as np

max_int = 2147483647
min_double = np.finfo(np.double).eps
max_iters_deviation = 1000


#==========================================================================
#===Array checks===========================================================
#==========================================================================
def check_nan_inf(x:np.ndarray, check_neg=False):
    if np.nan in x:
        raise ValueError("NaN in array")
    if np.inf in x:
        raise ValueError("inf in array")
    if check_neg and np.where(x<0)[0].__len__()>0:
        raise ValueError("array cant have negative values")
def check_arrays_compatible(X:np.ndarray,Y:np.ndarray=None,X_dot_product:np.ndarray=None,Y_dot_product:np.ndarray=None) -> tuple[np.ndarray]:
    """Returns the X and Y arrays so they are compatible for the distance calculation
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        An array where each row is a sample and each column is a feature.
        Optional. If None, it assumes `Y=X`.

    Returns
    -------
    X : ndarray version of X.

    Y : ndarray version of Y.
    """
    check_nan_inf(X)
    if Y==None:
        Y = X
    else:
        check_nan_inf(Y)
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Arrays must contain the same number of columns.")
    if X.dtype != Y.dtype:
        raise ValueError("Arrays must be of the same data type.")
    
    if X_dot_product is not None:
        if X_dot_product.shape[1] != X.shape[0]:
            raise ValueError("X_dot_products must be of shape (X.rows, 1).")
        if X_dot_product.dtype != X.dtype:
            raise ValueError("X_dot_products must be of the same data type as X.")
        check_nan_inf(X_dot_product)
    else:
        X_dot_product = (X*X).sum(axis=1)


    if Y_dot_product is not None:
        if Y_dot_product.shape[1] != Y.shape[0]:
            raise ValueError("Y_dot_products must be of shape (Y.rows, 1).")
        if Y_dot_product.dtype != Y.dtype:
            raise ValueError("Y_dot_products must be of the same data type as Y.")
        check_nan_inf(Y_dot_product)
    else:
        Y_dot_product = (Y * Y).sum(axis=1)
    
    return X,Y,X_dot_product,Y_dot_product
def check_arrays_compatible_2(X,Y=None) -> tuple[np.ndarray]:
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    check_nan_inf(X)
    if Y==None:
        Y = X
    else:
        check_nan_inf(Y)
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Arrays must contain the same number of columns.")
    if X.dtype != Y.dtype:
        raise ValueError("Arrays must be of the same data type.")
    return X,Y
#==========================================================================


#==========================================================================
#===Euclidean Distance=====================================================
#==========================================================================
def pairwise_euclidean_distance(X, *, sqrt=False, inf_diag=False) -> np.ndarray:
    """Compute the euclidean distances between the vectors of the given input.
    Parameters
    ----------
    X : {array-like, matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_X)
        Returns the distances between the row vectors of `X`.
    """
    check_nan_inf(X)

    X = np.array(X)

    result = __pairwise_euclidean_distance_fast(X)

    if sqrt:
        result = np.sqrt(result)
    if inf_diag:
        np.fill_diagonal(result, np.inf)
    
    return result
#--------------------------------------------------------------------------
def __pairwise_euclidean_distance(X:np.ndarray) -> np.ndarray:
    dist = np.zeros(shape=(X.shape[0],X.shape[0]))
    X_dot_product = (X*X).sum(axis=1)
    dist += X_dot_product.reshape(-1,1)
    dist -= 2 * np.dot(X,X.T)
    dist += X_dot_product.reshape(1,-1)

    return np.abs(dist)
def __pairwise_euclidean_distance_fast(X) -> np.ndarray:
    return np.sum((X[None, :] - X[:, None])**2, 2)
#==========================================================================


#==========================================================================
#===Joint Probabilities====================================================
#==========================================================================
#---T-Student--------------------------------------------------------------
def joint_probabilities_student(distances:np.ndarray)-> np.ndarray:
    """Obtain the joint probabilities (or affinities) of the points with the given distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points. The distances must be calculated without performing the square root

    perplexity : int. default = 10.0
        An array where each row is a sample and each column is a feature.
        Optional. If None, it assumes `Y=X`.
    
    tolerance : float. default = 0.1
        The ratio of tolerance for the goal perplexity expressed as
        per-unit (e.g.: a tolerance of 25% would be 0.25).
        Note: If 0, the result perplexity must be exact

    Returns
    -------
    probabilities : ndarray of shape (n_samples, n_samples) that contains the joint probabilities between the points given.
    """
    
    d1 = d2 = 1/(distances+1)

    d2 += 1e-8
    np.fill_diagonal(d2, 0.)
    
    result = d1/np.sum(d2)
    return np.maximum(result,0.)

#---Gaussian---------------------------------------------------------------
def joint_probabilities_gaussian(distances:np.ndarray, perplexity:int, tolerance:float=None) -> np.ndarray:
    """Obtain the joint probabilities (or affinities) of the points with the given distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points. The distances must be calculated without performing the square root

    perplexity : int. default = 10.0
        An array where each row is a sample and each column is a feature.
        Optional. If None, it assumes `Y=X`.
    
    tolerance : float. default = 0.1
        The ratio of tolerance for the goal perplexity expressed as
        per-unit (e.g.: a tolerance of 25% would be 0.25).
        Note: If 0, the result perplexity must be exact

    Returns
    -------
    probabilities : ndarray of shape (n_samples, n_samples) that contains the joint probabilities between the points given.
    """
    devs = search_deviations(distances,perplexity,tolerance)
    # cond_probs = np.maximum(conditional_p(distances, devs), 0.)
    cond_probs = conditional_p(distances, devs)
    result = (cond_probs+cond_probs.T)/(2*distances.shape[0])
    
    return result

def search_deviations(distances:np.ndarray, perplexity=10, tolerance=0.1, iters=1000) -> np.ndarray:
    """Obtain the Standard Deviations (σ) of each point in the given set (from the distances) such that
    the perplexities obtained when using them for the calculations of the conditional similarities will be
    within the given perplexity range.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points. The distances must be calculated without performing the square root

    perplexity : double. default = 10
        An array where each row is a sample and each column is a feature.
        Optional. If None, it assumes `Y=X`.
    
    tolerance : double. default = 0.1
        The ratio of tolerance for the goal perplexity expressed as
        per-unit (e.g.: a tolerance of 25% would be 0.25).
        Note: If 0, the result perplexity must be exact
    
    iters: int. default = 1000
        Maximum amount of iterations to calculate each deviation.

    Returns
    -------
    deviation : ndarray of shape (1, n_samples) of the Standard Deviations for each point.
    """
    result = []
    for i in range(distances.shape[0]):
        func = lambda dev: perplexity_from_conditional_p(conditional_p(distances[i:i+1, :], np.array([dev])))
        result.append(__search_deviation_indiv(func, perplexity, tolerance=tolerance, iters=iters))
    return np.array(result)
def __search_deviation_indiv(func, perplexity_goal, *, tolerance=1e-10, iters=1000, min_deviation=1e-20, max_deviation=10000.) -> float:
    for _ in range(iters):
        new_deviation = (min_deviation+max_deviation)/2.
        perplexity = func(new_deviation)
        if np.abs(perplexity - perplexity_goal) <= tolerance:
            return new_deviation
        if perplexity > perplexity_goal:
            max_deviation = new_deviation
        else:
            min_deviation = new_deviation
    return new_deviation
#==========================================================================


#==========================================================================
#===Conditional Probabilities==============================================
#==========================================================================
def conditional_p(distances:np.ndarray, deviations:np.ndarray) -> np.ndarray:
    """Compute the conditional similarities p_{j|i} and p_{i|j}
    using the distances and standard deviations 
    """

    aux1 = np.exp(-np.abs(distances)/(2*np.abs(deviations.reshape(-1,1))))
    aux2 = np.copy(aux1)

    np.fill_diagonal(aux2, 0.)
    #aux2+=1e-8


    result = aux1 / aux2.sum(axis=1).reshape([-1,1])
    return np.maximum(result, 0.)
#==========================================================================


#==========================================================================
#===Perplexity=============================================================
#==========================================================================
def perplexity_from_conditional_p(cond_p:np.ndarray) -> np.ndarray:
    """Compute the perplexity from the conditional p_{j|i} and p_{i|j}
    following the formula
    Perp(P) = 2**(-sum( p_{j|i}*log_2(p_{i|j})))
    """
    perp = -np.sum(cond_p*np.log2(cond_p),1)
    return 2**perp
#==========================================================================



#==========================================================================
#===========Usar distancia habiendo hecho raiz cuadrada====================
#==========================================================================
#-----------------------Vecinos mas cercanos-------------------------------
#¡¡¡¡¡¡¡¡¡¡¡¡¡¡CANDIDATO A OPTIMIZAR: nested for loop!!!!!!!!!!!!!!!!!!!!!!
def get_nearest_neighbors_indexes_by_distance(distances, k=None) -> np.ndarray:
    if distances.shape.__len__()!=2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square 2D array")
    
    
    result = np.ones_like(distances).astype(int)
    indices_sorted = np.argsort(distances)


    #recorrer los individuos
    for i in range(0, distances.shape[0]):
        for j in range(0, distances.shape[1]):
            ind_rank = indices_sorted[i][j]
            result[i][ind_rank] = j

    
    if k is not None: #si se especifica un limite de vecinos
        return result[:,:k]
    else:
        return result


def get_neighbor_ranking_by_distance_safe(distances) -> np.ndarray:
    if distances.shape.__len__()!=2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square 2D array")
    
    n = distances.shape[0]
    neighbors = distances.shape[1]

    result = np.ones_like(distances).astype(int)
    indices_sorted = np.argsort(distances)

    #recorrer los individuos
    for i in range(0, n):
        for rank in range(0, neighbors):
            j = indices_sorted[i][rank]
            result[i][j] = rank+1
    
    return result

def get_neighbor_ranking_by_distance_fast(distances) -> np.ndarray:
    if distances.shape.__len__()!=2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square 2D array")

    result = np.ones_like(distances).astype(int)
    indices_sorted = np.argsort(distances)

    filas, columnas = np.indices(distances.shape)

    result[filas, indices_sorted] += columnas
    
    return result

