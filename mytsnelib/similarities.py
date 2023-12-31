import numpy as np

max_int = 2147483647
min_double = np.finfo(np.double).eps
max_iters_deviation = 1000

def euclidean_distance(X:np.ndarray, Y:np.ndarray=None, X_dot_products:np.ndarray=None, Y_dot_products:np.ndarray=None, return_squared=False):
    """Compute the euclidean distances between the vectors of X and Y.
    
    
    Parameters
    ----------
    X : {array-like, matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, matrix} of shape (n_samples_Y, n_features), \
            default=None
        An array where each row is a sample and each column is a feature.
        If none is given, method uses `Y=X`.

    Y_norm_squared : array-like of shape (n_samples_Y,) or (n_samples_Y, 1) \
            or (1, n_samples_Y), default=None
        Pre-computed dot-products of vectors in Y

    squared : bool, default=False
        If True, returns squared euclidean distances.

    X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1) \
            or (1, n_samples_X), default=None
        Pre-computed dot-products of vectors in X.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.
    """
    X, Y = check_arrays_compatible(X,Y)
    if X_dot_products is not None:
        if X_dot_products.shape[1] != X.shape[0]:
            raise ValueError("X_dot_products must be of shape (X.rows, 1).")
        if X_dot_products.dtype != X.dtype:
            raise ValueError("X_dot_products must be of the same data type as X.")
    else:
        X_dot_products = (X*X).sum(axis=1)


    if Y_dot_products is not None:
        if Y_dot_products.shape[1] != Y.shape[0]:
            raise ValueError("Y_dot_products must be of shape (Y.rows, 1).")
        if Y_dot_products.dtype != Y.dtype:
            raise ValueError("Y_dot_products must be of the same data type as Y.")
    else:
        Y_dot_products = (Y * Y).sum(axis=1)
    
    if np.nan in X_dot_products:
        raise ValueError("X_dot_products contains NaN")
    if np.inf in X_dot_products:
        raise ValueError("X_dot_products contains inf")
    
    if np.nan in Y_dot_products:
        raise ValueError("Y_dot_products contains NaN")
    if np.inf in Y_dot_products:
        raise ValueError("Y_dot_products contains inf")
    
    
    result = __euclidean_distance(X,Y,X_dot_products,Y_dot_products,return_squared)
    
    if np.nan in result:
        raise ValueError("Distance contains NaN, wrong metric")
    if np.inf in result:
        raise ValueError("Distance can't be infinite, wrong metric")
    if np.where(result<0)[0].__len__()>0:
        raise ValueError("Distance can't be negative, wrong metric")

    return result


def __euclidean_distance(X, Y, X_dot_products, Y_dot_products, return_squared):
    """This is where the euclidean distances are calculated.
    """
    XX = X_dot_products.reshape(-1,1)
    YY = Y_dot_products.reshape(1,-1)

    dist = np.zeros(shape=(X.shape[0],X.shape[0]))
    
    dist += XX
    dist -= 2 * np.dot(X,Y.T)
    dist += YY

    dist = np.abs(dist)
    return dist if not return_squared else np.sqrt(dist)

def euclidean_distance_neighbors(X:np.ndarray, Y:np.ndarray=None, X_dot_products:np.ndarray=None, Y_dot_products:np.ndarray=None, return_squared=False, n_neighbors=10):
    """Same as the euclidean_distance method, but replaces with np.inf every distance
    except those corresponding to the nearest n_neighbors.
    
    Parameters
    ----------
    X : {array-like, matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, matrix} of shape (n_samples_Y, n_features), \
            default=None
        An array where each row is a sample and each column is a feature.
        If none is given, method uses `Y=X`.

    Y_norm_squared : array-like of shape (n_samples_Y,) or (n_samples_Y, 1) \
            or (1, n_samples_Y), default=None
        Pre-computed dot-products of vectors in Y

    squared : bool, default=False
        If True, returns squared euclidean distances.

    X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1) \
            or (1, n_samples_X), default=None
        Pre-computed dot-products of vectors in X.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y` with the (n_samples_X-n_neighbors)
        farthest neighbors replaced with np.inf.
    """
    distances = euclidean_distance(X,Y, X_dot_products, Y_dot_products, return_squared)
    
    nearest_index = find_nearest_neighbors_index(distances,n_neighbors)
    n = distances.shape[0]
    max_dist = np.max(distances)
    for i in range(n):
        for j in range(n):
            if j not in nearest_index[i] or (i==j and (Y is None or np.array_equal(X,Y))):
                distances[i][j] = 10 *max_dist

    return distances

def check_arrays_compatible(X:np.ndarray,Y:np.ndarray=None):
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
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if Y==None:
        Y = X
    elif not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
    
    if np.nan in X:
        raise ValueError("X contains NaN")
    if np.inf in X:
        raise ValueError("X contains inf")
    
    if np.nan in Y:
        raise ValueError("Y contains NaN")
    if np.inf in Y:
        raise ValueError("Y contains inf")
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Arrays must contain the same number of columns.")
    if X.dtype != Y.dtype:
        raise ValueError("Arrays must be of the same data type.")
    return X,Y

def conditional_p(distances:np.ndarray, deviations:np.ndarray):
    """Compute the conditional similarities p_{j|i} and p_{i|j}
    using the distances and standard deviations 
    """
    probs = np.exp(-distances/2*np.square(deviations.reshape(-1,1)))
    
    probs += min_double
    result = probs/np.sum(probs, axis=1)

    return result

def perplexity_from_conditional_p(cond_p:np.ndarray):
    """Compute the perplexity from the conditional p_{j|i} and p_{i|j}
    following the formula
    Perp(P) = 2**( -sum( p_{j|i}* log_2(p_{i|j}) ) )
    """
    perp = -np.sum(cond_p*np.log2(cond_p),1)
    return 2**perp

def search_deviations(distances:np.ndarray, perplexity=10, tolerance=0.1):
    """Obtain the Standard Deviations (σ) of each point in the given set (from the distances) such that
    the perplexities obtained when using them for the calculations of the conditional similarities will be
    within the given perplexity range.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points.

    perplexity : double. default = 10
        An array where each row is a sample and each column is a feature.
        Optional. If None, it assumes `Y=X`.
    
    tolerance : double. default = 0.1
        The ratio of tolerance for the goal perplexity expressed as
        per-unit (e.g.: a tolerance of 25% would be 0.25).
        Note: If 0, the result perplexity must be exact

    Returns
    -------
    deviation : ndarray of shape (1, n_samples) of the Standard Deviations for each point.
    """
    max_perp = (1+tolerance) * perplexity
    min_perp = max(min_double, (1-tolerance) * perplexity) 
    

    n = distances.shape[0]
    max_deviation = 1000 * np.ones(shape=(n,))
    min_deviation = min_double*np.ones(shape=(n,)) + min_double

    while True:

        
        computed_deviation = (max_deviation+min_deviation)/2

        cond_p = conditional_p(distances,computed_deviation)

        perplexities = perplexity_from_conditional_p(cond_p)

        max_dev_temp = max_deviation
        min_dev_temp = min_deviation

        max_deviation = np.where(perplexities>=min_perp, computed_deviation, max_dev_temp)
        min_deviation = np.where(perplexities<=max_perp, computed_deviation, min_dev_temp)
        
        if np.array_equal(min_deviation, max_deviation) or max((max_deviation-min_deviation)/min_deviation)<max(tolerance, 0.001):break

    return computed_deviation

def conditional_probabilities_from_distances(distances:np.ndarray, perplexity:int, tolerance:float=None):
    """Obtain the conditional probabilities of a set of distances

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points.

    perplexity : int. default = 10
        An array where each row is a sample and each column is a feature.
        Optional. If None, it assumes `Y=X`.
    
    tolerance : double. default = 0.1
        The ratio of tolerance for the goal perplexity expressed as
        per-unit (e.g.: a tolerance of 25% would be 0.25).
        Note: If 0, the result perplexity must be exact.
    distribution: str. default = 'gaussian'
        The type of distribution to use to compute the conditional probabilities.

    distribution_params: Optional, dict[Type]
        A dictionary of any further parameters required for the distribution type selected.
        In the case of t-student, must be the number of degrees of freedom, and Type must be int. 

    Returns
    -------
    probabilities : ndarray of shape (n_samples, n_samples) that contains the conditional probabilities between the points given.
    """
    
    devs = search_deviations(distances,perplexity,tolerance)
    return conditional_p(distances, devs)

def joint_probabilities(distances:np.ndarray, perplexity:int, tolerance:float=None, distribution='gaussian'):
    """Obtain the joint probabilities (or affinities) of the points with the given distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        An array with the distances between the different points.

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
    accepted_distr=['gaussian', 't-student']
    if distribution not in accepted_distr:
        raise ValueError("Only distributions supported are gaussian and t-student")
    elif distribution=='gaussian':
        n = distances.shape[0]
        cond_probs = conditional_probabilities_from_distances(distances,perplexity,tolerance)
        # np.fill_diagonal(cond_probs, min_double)
        result = (cond_probs+cond_probs.T)/(2*n)
    
    elif distribution=='t-student':
        dist = np.ones_like(distances) + distances
        result = 2. * np.sum(1/dist)
        result = 1/(dist*result)
    
    
    return np.where(result>0.,result,min_double) 
        

def find_nearest_neighbors_index(distances:np.ndarray, n_neighbors:int):
    """Find the indexes of the nearest n_neighbors.
    
    Parameters  
    -------
    distances: ndarray of shape (n_samples, n_samples).
        The distances of the dataset.
    
    n_neighbors: int
        The amount of neighbors to return for each point.
    
    Returns
    -------
    indexes: ndarray of shape (n_samples, n_neighbors).
        The indexes of the nearest n_neighbors.
    """

    n = distances.shape[0]
    

    result = np.zeros(shape=(n, n_neighbors))
    for i in range(n):
        indexes = np.array([])
        dist = distances[i]
        
        dists_sorted = np.sort(dist[dist>0])
        unique_dists_sorted = np.unique(dists_sorted)
        for d in unique_dists_sorted:
            ind = np.where(dist == d)[0]
            indexes = np.append(indexes, ind)
            if indexes.shape[0]>n_neighbors:
                indexes = indexes[:n_neighbors]
                break
        
        result[i] = indexes
    return result

def get_neighbors_ranked_by_distance(distances):
    if distances.shape.__len__()!=2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square 2D array")
    result = np.ones_like(distances).astype(int)
    for i in range(0, distances.shape[0]):
        index_sorted = np.argsort(distances[i])
        for j in range(0, distances.shape[1]):
            ind_rank = index_sorted[j]
            result[i][ind_rank] = j+1
        result[i] = index_sorted
    return result