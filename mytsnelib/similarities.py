import numpy as np
import gc
#===Euclidean Distance=====================================================
def pairwise_euclidean_distance(X, *, sqrt=False) -> np.ndarray:
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
    X = np.nan_to_num(X)
    espacio_maybe = len(X)*len(X)*X.dtype.itemsize
    if espacio_maybe>9e9:
        print("Espacio que iba a ocupar: {} GB".format(espacio_maybe/1e9))
        raise ValueError("No")
    if espacio_maybe*X.shape[1]<7e9:
        # rapido pero consume un cristo y medio de memoria
        aux = np.sum(np.square(np.expand_dims(X,0) - np.expand_dims(X,1)), 2)
        result = np.sqrt(aux) if sqrt else np.abs(aux)
        del aux
    else:
        X_cuadrado = np.sum(np.square(X), axis=1)
        Y_cuadrado = np.reshape(X_cuadrado, [-1,1])
        producto = np.dot(X, X.T)
        aux = X_cuadrado - 2*producto + Y_cuadrado
        result = np.sqrt(aux) if sqrt else np.abs(aux)
        del aux,producto,X_cuadrado, Y_cuadrado
    
    gc.collect()
    
    return result



#===Joint Probabilities (Gaussian))========================================
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
    dists = np.abs(distances)
    
    devs = search_deviations(dists,perplexity,tolerance)
    cond_probs = __conditional_p(dists, devs)
    
    result = (cond_probs+cond_probs.T)/(2*dists.shape[0])
    return result
#---Deviations-------------------------------------------------------------
def search_deviations(distances:np.ndarray, perplexity=10., tolerance=0.1, iters=1000) -> np.ndarray:
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
    result = np.zeros(len(distances))
    for i in range(len(distances)):
        func = lambda dev: __perplexity(__conditional_p(distances[i:i+1, :], np.array([dev])))
        result[i] = __search_deviation_indiv(func, perplexity, tolerance, iters)
    return result
def __search_deviation_indiv(func, perplexity_goal, tolerance, iters, *, min_deviation=1e-20, max_deviation=10000.) -> float:
    for _ in range(iters):
        new_deviation = np.mean([min_deviation, max_deviation])
        diff = func(new_deviation) - perplexity_goal
        if diff > 0:
            max_deviation = new_deviation
        else:
            min_deviation = new_deviation
        
        if abs(diff) <= tolerance:
            return new_deviation
    return new_deviation

#---Perplexity-------------------------------------------------------------
def __perplexity(cond_p:np.ndarray) -> np.ndarray:
    """Compute the perplexity from all the conditional p_{j|i}
    following the formula
    Perp(P_i) = 2**(-sum( p_{j|i}*log_2(p_{j|i})))
    """
    
    entropy = -np.sum(cond_p*np.nan_to_num(np.log2(cond_p)), 1)
    result = 2**entropy

    del entropy
    return result
#---Conditional Probabilities----------------------------------------------
def __conditional_p(distances:np.ndarray, sigmas:np.ndarray) -> np.ndarray:
    """Compute the conditional similarities p_{j|i} and p_{i|j}
    using the distances and standard deviations 
    """
    
    d1 = np.exp(-distances/(2*np.square(np.reshape(sigmas, [-1,1]))))
    d2 = np.copy(d1)
    np.fill_diagonal(d2, 0)
    # d1+=1e-8
    result = d1 / np.reshape(np.sum(d2, axis=1), [-1,1])

    result[result<=0] = 0
    return result

#===Joint Probabilities (T-Student)========================================
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
    d1 = 1/(1+distances)
    d2 = np.copy(d1)

    np.fill_diagonal(d2, 0)
    
    result = d1/np.sum(d2)

    result[result<0] = 0

    return result

#===Vecinos mas cercanos===================================================
def get_neighbor_ranking_by_distance_safe(distances) -> np.ndarray:
    if distances.shape.ndim!=2 or len(distances) != distances.shape[1]:
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
    if distances.shape.ndim!=2 or len(distances) != distances.shape[1]:
        raise ValueError("distances must be a square 2D array")

    result = np.ones_like(distances).astype(int)
    indices_sorted = np.argsort(distances)

    filas, columnas = np.indices(distances.shape)

    result[filas, indices_sorted] += columnas
    
    return result
