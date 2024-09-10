import numpy as np
#===Euclidean Distance=====================================================
def pairwise_euclidean_distance(X, *, sqrt=False, condensed=False) -> np.ndarray:
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
    from scipy.spatial import distance
    metrica = "euclidean" if sqrt else "sqeuclidean"
    result = distance.pdist(X, metric=metrica)
    if not condensed:
        result = distance.squareform(result)
    return result

#===Joint Probabilities (Gaussian))========================================
def joint_probabilities_gaussian(dists:np.ndarray, perplexity:int, tolerance:float=None, search_iters=1000) -> np.ndarray:
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
    n = len(dists)
    not_diag = ~np.eye(dists.shape[0], dtype=bool)
    cond_probs = np.zeros_like(dists, dtype=float)
    for i in range(n):
        cond_probs[i] = __search_cond_p(dists[i:i+1, :], perplexity, tolerance, search_iters, not_diag[i:i+1,:])
    result = (cond_probs+cond_probs.T)/(2*n)
    del cond_probs
    return result
#---Deviations-------------------------------------------------------------
def __search_cond_p(dist, perplexity_goal, tolerance, iters, not_diag, *, min_deviation=1e-20, max_deviation=10000.) -> float:
    for _ in range(iters):
        new_deviation = np.mean([min_deviation, max_deviation])
        p_ = __conditional_p(dist, np.array([new_deviation]), not_diag)
        diff = __perplexity(p_) - perplexity_goal
        if diff > 0: # nueva_perplejidad > objetivo
            max_deviation = new_deviation
        else: # nueva_perp < objetivo
            min_deviation = new_deviation
        if abs(diff) <= tolerance:
            return p_[0]
    return p_[0]

#---Perplexity-------------------------------------------------------------
def __perplexity(cond_p:np.ndarray) -> np.ndarray:
    """Compute the perplexity from all the conditional p_{j|i}
    following the formula
    Perp(P_i) = 2**(-sum( p_{j|i}*log_2(p_{j|i})))
    """
    return 2**(-np.sum(cond_p*np.nan_to_num(np.log2(cond_p)), 1))
#---Conditional Probabilities----------------------------------------------
def __conditional_p(distances:np.ndarray, sigmas:np.ndarray, not_diag) -> np.ndarray:
    """Compute the conditional similarities p_{j|i} and p_{i|j}
    using the distances and standard deviations 
    """
    aux = np.exp(-distances/(2*np.square(sigmas.reshape([-1,1]))))
    return aux / aux.sum(axis=1, where=not_diag).reshape([-1,1])

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
    d = 1/(1+distances)
    return d/d.sum(where=~np.eye(distances.shape[0], dtype=bool))

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
