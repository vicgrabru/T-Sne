import numpy as np
#===Euclidean Distance=====================================================
def pairwise_euclidean_distance(X, *, sqrt=False, condensed=False) -> np.ndarray:
    """Compute the euclidean distances between the vectors of the given input.
    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.
    
    sqrt : boolean, default=False.
        If True, uses metric "sqeuclidean".
        Uses "euclidean" otherwise.
    
    condensed : bool, default=False
        If True, returns the condensed form of the array.
        Returns square form otherwise.
    
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
def joint_probabilities_gaussian(dists:np.ndarray, perplexity:int, tolerance:float=0., search_iters=10000) -> np.ndarray:
    """Obtain the joint probabilities (or affinities) of the points with the given distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points. The distances must be calculated without performing the square root

    perplexity : float, default = 10.0
        Goal perplexity value.
    
    tolerance : float, default = 0.1
        Acceptable perplexities will be in the range [perplexity-tolerance, perplexity+tolerande].
        Note: If 0, the result perplexity must be exact
    
    search_iters : int, default = 10000
        Number of iterations of search for the value of each sigma

    Returns
    -------
    probabilities : ndarray of shape (n_samples, n_samples) that contains the joint probabilities between the points given.
    """
    n = dists.shape[0]
    not_diag = ~np.eye(n, dtype=bool)
    cond_probs = np.zeros_like(dists, dtype=np.float64)
    for i in range(n):
        cond_probs[i] = __search_cond_p(dists[i:i+1,:], perplexity, tolerance, search_iters, not_diag[i:i+1,:])
    return (cond_probs+cond_probs.T)/(2*n)

#Deviations
def __search_cond_p(dist, goal, tolerance, iters, not_diag, *, min_deviation=1e-20, max_deviation=1e5) -> float:
    i = 0
    while True:
        new_deviation = (min_deviation+max_deviation)/2
        p = __conditional_p(dist, [new_deviation], not_diag)
        
        diff = __perplexity(p) - goal
        cond = tolerance==0 and i>=iters
        if abs(diff) <= abs(tolerance) or cond:
            break

        if diff > 0:
            max_deviation = new_deviation
        else:
            min_deviation = new_deviation
        i+=1
    return p[0]

#Perplexity
def __perplexity(cond_p:np.ndarray) -> np.ndarray:
    condition = cond_p!=0
    ax = None if min(cond_p.shape)==1 else 1
    return 2**(-np.sum(cond_p*np.log2(cond_p, where=condition), axis=ax, where=condition))

#Conditional Probabilities
def __conditional_p(distances:np.ndarray, sigmas, not_diag) -> np.ndarray:
    aux = np.exp(-distances/(2*np.square(np.reshape(sigmas, [-1,1]))))
    if min(distances.shape)==1:
        indice = np.argmin(distances)
        aux[0][indice]=0.
    else:
        np.fill_diagonal(aux, 0.)
    return aux / aux.sum(axis=1, where=not_diag)

#===Joint Probabilities (T-Student)========================================
def joint_probabilities_student(distances:np.ndarray)-> np.ndarray:
    """Obtain the joint probabilities q.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points. The distances must be calculated without performing the square root

    Returns
    -------
    probabilities : ndarray of shape (n_samples, n_samples) that contains the joint probabilities between the points given.
    """
    d = 1/(1+distances)
    return d/(d.sum()-d.trace())

#===Nearest Neighbors===================================================
def __get_neighbor_ranking_by_distance_safe(distances) -> np.ndarray:
    if distances.shape.ndim!=2 or len(distances) != distances.shape[1]:
        raise ValueError("distances must be a square 2D array")
    
    n = distances.shape[0]
    neighbors = distances.shape[1]

    result = np.ones_like(distances).astype(int)
    indices_sorted = np.argsort(distances, axis=1)

    #recorrer los individuos
    for i in range(n):
        for rank in range(neighbors):
            j = indices_sorted[i][rank]
            result[i][j] = rank+1
    return result

def __get_neighbor_ranking_by_distance_fast(distances) -> np.ndarray:
    if distances.shape.ndim!=2 or len(distances) != distances.shape[1]:
        raise ValueError("distances must be a square 2D array")

    result = np.ones_like(distances).astype(int)
    indices_sorted = np.argsort(distances, axis=1)

    filas, columnas = np.indices(distances.shape)

    result[filas, indices_sorted] += columnas
    
    return result

def __get_is_neighbor(distances:np.ndarray, n_neighbors):
    result = np.zeros(shape=distances.shape, dtype=bool)
    filas = np.array([1 for _ in range(len(n_neighbors))], dtype=int)
    indices_neighbors = np.argsort(distances, axis=1)[:,:n_neighbors]
    for i in range(len(result)):
        result[filas*i, indices_neighbors[i]] = True
    return result
