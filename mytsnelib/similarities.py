import numpy as np

max_int = 2147483647
min_double = np.finfo(np.double).eps
max_iters_deviation = 100000

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
    
    return __euclidean_distance(X,Y,X_dot_products,Y_dot_products,return_squared)


def __euclidean_distance(X, Y, X_dot_products, Y_dot_products, return_squared):
    """This is where the euclidean distances are calculated.
    """
    XX = X_dot_products.reshape(-1,1)
    YY = Y_dot_products.reshape(1,-1)

    # dist(X,Y) = sqrt ( sum_i(X_i^2 - 2*X_i*Y_i + Y_i^2) )

    #X, Y son 2 matrices de vectores: dist para cada X con cada Y
    #X_i*Y_i

    #dist(i,j) = sqrt (sum_l(Xi[l]*Yj[l])
    dist = -2 * np.dot(X,Y.T)
    dist += XX
    dist += YY

    
    return dist if not return_squared else np.sqrt(dist, out=dist)

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
    print("neighbors used for the distances: {}".format(n_neighbors))
    """
    print("=======================")
    print("metodo: euclidean_distances_neighbors")
    print("printed: distances")
    print("-----------------------")
    print(distances)
    print("=======================")
    """
    
    nearest_index = find_nearest_neighbors_index(distances,n_neighbors)
    n = distances.shape[0]
    max_dist = np.max(distances)
    for i in range(0, n):
        for j in range(0, n):
            if j not in nearest_index[i] or (np.array_equal(X,Y) or Y is None) and i==j:
                distances[i][j] = max_dist
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
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Arrays must contain the same number of columns.")
    if X.dtype != Y.dtype:
        raise ValueError("Arrays must be of the same data type.")
    return X,Y


def conditional_p(distances:np.ndarray, deviations:np.ndarray):
    """Compute the conditional similarities p_{j|i} and p_{i|j}
    using the distances and standard deviations 
    """
    """
    print("======================")
    print("metodo: conditional_p")
    print("printed: distances")
    print("----------------------")
    print(distances)
    print("======================")
    """

    probs = np.exp(-distances/2*np.square(deviations.reshape((-1,1))))
    np.fill_diagonal(probs, 0.)
    probs+= 1e-8
    return probs/probs.sum(axis=1).reshape([-1,1])


def perplexity_from_conditional_p(cond_p:np.ndarray):
    """Compute the perplexity from the conditional p_{j|i} and p_{i|j}
    following the formula
    Perp(P) = 2**( -sum( p_{j|i}* log_2(p_{i|j}) ) )
    """
    """
    print("=====================================")
    print("metodo: perplexity_from_conditional_p")
    print("-------------------------------------")
    print("printing: cond_p")
    print(cond_p)
    print("-------------------------------------")
    print("printing: cond_p.T")
    print(cond_p.T)
    print("=====================================")
    """
    perp = -np.sum(cond_p*np.log2(cond_p.T),1)
    return 2**perp


def search_deviations(distances:np.ndarray, perplexity=10., tolerance=0.1):
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

    """
    print("======================")
    print("metodo: search_deviations")
    print("printed: distances")
    print("----------------------")
    print(distances)
    print("----------------------")
    print("printed: distances.shape")
    print(distances.shape)
    print("======================")
    """
    
    max_perp = (1.+tolerance) * perplexity
    min_perp = max(min_double, (1.-tolerance) * perplexity) 
    
    
    n = distances.shape[0]
    max_deviation = 100000000000000. * np.ones(shape=(n,))
    min_deviation = min_double*np.ones(shape=(n,)) + min_double
    #computed_deviation = 0.5*(max_deviation + min_deviation)

    for _ in range(0, max_iters_deviation):
        temp_devs = 0.5*(max_deviation+min_deviation)
        computed_deviation = np.where(max_deviation==min_deviation, max_deviation, temp_devs)

        perplexities = perplexity_from_conditional_p(conditional_p(distances,computed_deviation))

        for i in range(0,n):
            if perplexities[i] >= min_perp: max_deviation[i] = computed_deviation[i]
            if perplexities[i] <= max_perp: min_deviation[i] = computed_deviation[i]
        
        if np.array_equal(min_deviation, max_deviation): break
    
    return computed_deviation
    #return __search_deviation(distances,min_perp,max_perp, min_initial_deviation, max_initial_deviation)

"""
def __search_deviation(distances, min_perplexity, max_perplexity, min_current_deviation, max_current_deviation):
    n = distances.shape[0]
    temp_devs = np.zeros(shape=(n,))

    for i in range(0, n):
        max_current = max_current_deviation[i]
        min_current = min_current_deviation[i]
        
        if np.array_equal(max_current, min_current):
            temp_devs[i] = max_current
        else:
            computed_deviation = min_current + max_current
            computed_deviation *= 0.5
            temp_devs[i] = computed_deviation
            
    
    perplexities = perplexity_from_conditional_p(conditional_p(distances,temp_devs))
    
    for i in range(0,n):
        if perplexities[i] >= min_perplexity: max_current_deviation[i] = temp_devs[i]
        if perplexities[i] <= max_perplexity: min_current_deviation[i] = temp_devs[i]
    
    if np.array_equal(min_current_deviation, max_current_deviation):
        return min_current_deviation
    else:
        return __search_deviation(distances, min_perplexity, max_perplexity, min_current_deviation, max_current_deviation)

"""


def conditional_probabilities_from_distances(distances:np.ndarray, perplexity:int, tolerance:float=None, distribution='gaussian', distribution_params=None):
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

    """
    print("======================")
    print("metodo: conditional_probabilities_from_distances")
    print("printed: distances")
    print("----------------------")
    print(distances)
    print("======================")
    """


    if distribution=='gaussian':
        devs = search_deviations(distances,perplexity,tolerance)
        result = conditional_p(distances, devs)
    elif distribution=='t-student':
        if distribution_params==None or not isinstance(distribution_params, np.ndarray) or distribution_params.ndtype==np.int_:
            raise ValueError("Degrees of freedom required for the t-student distribution")
        else:
            distances /= distribution_params[0]
            distances += 1.
            distances **= (distribution_params[0] + 1.)/-2.
            return distances/(2. * np.sum(distances))
    else:
        raise ValueError("Only distributions supported are gaussian and t-student")
    return result


def joint_probabilities(distances:np.ndarray, perplexity:int, tolerance:float=None):
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
    """
    print("======================")
    print("metodo: joint_probabilities")
    print("printed: distances")
    print("----------------------")
    print(distances)
    print("======================")
    """
    cond_probs = conditional_probabilities_from_distances(distances,perplexity,tolerance)

    probs_sum = cond_probs + cond_probs.T
    denom = 2.*cond_probs.shape[0]

    return probs_sum/denom

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
    """
    print("======================")
    print("metodo: find_nearest_neighbors_index")
    print("printed: distances")
    print("----------------------")
    print(distances)
    print("======================")
    """
    
    n = distances.shape[0]
    result = np.zeros(shape=(n, n_neighbors))
    for i in range(0, n):
        indexes = np.array([])
        dist = distances[i]
        
        dists_sorted = np.sort(dist[dist>0])
        unique_dists_sorted = np.unique(dists_sorted)
        for d in unique_dists_sorted:
            ind = np.where(dist == d)[0]
            indexes = np.append(indexes, ind)
            if indexes.__len__()>n_neighbors:
                indexes = indexes[0:n_neighbors]
                break
        result[i] = indexes
    return result