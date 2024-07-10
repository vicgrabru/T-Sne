import numpy as np

max_int = 2147483647
min_double = np.finfo(np.double).eps
max_iters_deviation = 1000

def check_nan_inf(x:np.ndarray, check_neg=False):
    if np.nan in x:
        raise ValueError("NaN in array")
    if np.inf in x:
        raise ValueError("inf in array")
    if check_neg and np.where(x<0)[0].__len__()>0:
        raise ValueError("array cant have negative values")

def check_arrays_compatible(X:np.ndarray,Y:np.ndarray=None,X_dot_product:np.ndarray=None,Y_dot_product:np.ndarray=None):
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
    else:
        X_dot_product = (X*X).sum(axis=1)

    check_nan_inf(X_dot_product)

    if Y_dot_product is not None:
        if Y_dot_product.shape[1] != Y.shape[0]:
            raise ValueError("Y_dot_products must be of shape (Y.rows, 1).")
        if Y_dot_product.dtype != Y.dtype:
            raise ValueError("Y_dot_products must be of the same data type as Y.")
    else:
        Y_dot_product = (Y * Y).sum(axis=1)
    
    check_nan_inf(Y_dot_product)
    

    return X,Y,X_dot_product,Y_dot_product

def euclidean_distance(X:np.ndarray, Y:np.ndarray=None, X_dot_product:np.ndarray=None, Y_dot_product:np.ndarray=None):
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

    return_square_rooted : bool, default=False
        If True, returns square rooted euclidean distances.

    X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1) \
            or (1, n_samples_X), default=None
        Pre-computed dot-products of vectors in X.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.
    """
    X, Y, X_dot_product, Y_dot_product = check_arrays_compatible(X,Y,X_dot_product,Y_dot_product)
    
    
    # result = __euclidean_distance_1(X,Y,X_dot_products,Y_dot_products)
    result = __euclidean_distance_2(X,Y)
    check_nan_inf(result,True)

    return result


def __euclidean_distance_1(X, Y, X_dot_products, Y_dot_products):
    """This is where the euclidean distances are calculated.
    """
    XX = X_dot_products.reshape(-1,1)
    YY = Y_dot_products.reshape(1,-1)

    dist = np.zeros(shape=(X.shape[0],X.shape[0]))
    
    dist += XX
    dist -= 2 * np.dot(X,Y.T)
    dist += YY

    dist = np.abs(dist)
    return np.sqrt(dist)

def __euclidean_distance_2(X,Y):
    result = np.zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(0,X.shape[0]):
        for j in range(0, Y.shape[0]):
            aux = 0.0
            for k in range(0, X.shape[1]):
                aux += np.power(X[i][k]-Y[j][k],2)
            
            result[i][j] = np.sqrt(aux)
    return result



def euclidean_distance_neighbors(X:np.ndarray, Y:np.ndarray=None, X_dot_products:np.ndarray=None, Y_dot_products:np.ndarray=None, n_neighbors=10):
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
    distances = euclidean_distance(X,Y, X_dot_products, Y_dot_products)
    
    nearest_index = find_nearest_neighbors_index(distances,n_neighbors)
    n = distances.shape[0]
    max_dist = np.max(distances)
    for i in range(n):
        for j in range(n):
            if j not in nearest_index[i] or (i==j and (Y is None or np.array_equal(X,Y))):
                distances[i][j] = 10 *max_dist

    return distances



def conditional_p(distances:np.ndarray, deviations:np.ndarray):
    """Compute the conditional similarities p_{j|i} and p_{i|j}
    using the distances and standard deviations 
    """
    probs = np.exp(-0.5*np.square(distances)/np.square(deviations.reshape(-1,1)))
    
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

def search_deviations_exact(distances:np.ndarray, perplexity=10, iters=1000):
    #max_perp = (1+tolerance) * perplexity
    #min_perp = max(min_double, (1-tolerance) * perplexity) 
    
    i = 0
    n = distances.shape[0]
    max_deviation = 1000 * np.ones(shape=(n,))
    min_deviation = min_double*np.ones(shape=(n,)) + min_double

    # while not np.array_equal(min_deviation, max_deviation) and max(np.divide((max_deviation-min_deviation),min_deviation))>=tolerance:
    while i != iters and not np.array_equal(min_deviation, max_deviation):
        computed_deviation = (max_deviation+min_deviation)/2

        cond_p = conditional_p(distances,computed_deviation)

        perplexities = perplexity_from_conditional_p(cond_p)

        

        for i in range(0,n):
            if perplexities[i]>perplexity:
                max_deviation[i] = computed_deviation[i]

            if perplexities[i]<perplexity:
                min_deviation[i] = computed_deviation[i]
        
        i+=1
        # max_dev_temp = max_deviation
        # min_dev_temp = min_deviation
        # max_deviation = np.where(perplexities>=min_perp, computed_deviation, max_dev_temp)
        # min_deviation = np.where(perplexities<=max_perp, computed_deviation, min_dev_temp)
        

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
    
    #devs = search_deviations(distances,perplexity,tolerance)
    devs = search_deviations_exact(distances,perplexity)
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
        result = (cond_probs+cond_probs.T)/(2*n)
    
    elif distribution=='t-student':
        aux = np.power(distances,2) + 1
        d1 = np.power(aux, -1)
        d2 = np.copy(d1)
        np.fill_diagonal(d2, 0.)
        result = d1/np.sum(d2)
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
    #recorrer los individuos
    for i in range(0, distances.shape[0]):
        index_sorted = np.argsort(distances[i])
        for j in range(0, distances.shape[1]):
            ind_rank = index_sorted[j]
            result[i][ind_rank] = j+1
        result[i] = index_sorted
    return result