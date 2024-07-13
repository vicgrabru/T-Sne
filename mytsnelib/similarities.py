import numpy as np

max_int = 2147483647
min_double = np.finfo(np.double).eps
max_iters_deviation = 1000


#==========================================================================
#=============Comprobaciones de compatibilidad=============================
#==========================================================================


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


def check_arrays_compatible_2(X,Y=None):
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
#====Calculan la distancia y la devuelven sin hacer la raiz cuadrada=======
#==========================================================================


def pairwise_euclidean_distance(X):
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

    return __pairwise_euclidean_distance_fast(X)
    

def __pairwise_euclidean_distance(X:np.ndarray):
    dist = np.zeros(shape=(X.shape[0],X.shape[0]))
    X_dot_product = (X*X).sum(axis=1)
    dist += X_dot_product.reshape(-1,1)
    dist -= 2 * np.dot(X,X.T)
    dist += X_dot_product.reshape(1,-1)

    return np.abs(dist)
def __pairwise_euclidean_distance_fast(X):
    return np.sum((X[None, :] - X[:, None])**2, 2)



#==========================================================================
#===========Usar distancia sin hacer la raiz cuadrada======================
#==========================================================================
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
        aux = distances + 1
        d1 = 1/aux
        d2 = 1/aux
        np.fill_diagonal(d2, 0.)
        d2 += 1e-8
        result = d1/np.sum(d2)
    return np.where(result>0.,result,min_double)

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
    
    devs = search_deviations_1(distances,perplexity,tolerance)
    #devs = search_deviations_2(distances,perplexity,tolerance)
    #devs = search_deviations_exact(distances,perplexity)
    return conditional_p(distances, devs)

def conditional_p(distances:np.ndarray, deviations:np.ndarray):
    """Compute the conditional similarities p_{j|i} and p_{i|j}
    using the distances and standard deviations 
    """

    aux1 = np.exp(-np.abs(distances)/(2*np.abs(deviations.reshape(-1,1))))
    aux2 = np.copy(aux1)

    np.fill_diagonal(aux2, 0.)
    aux2+=1e-8
    
    result = aux1 / aux2.sum(axis=1).reshape([-1,1])
    return result



def perplexity_from_conditional_p(cond_p:np.ndarray):
    """Compute the perplexity from the conditional p_{j|i} and p_{i|j}
    following the formula
    Perp(P) = 2**( -sum( p_{j|i}* log_2(p_{i|j}) ) )
    """
    perp = -np.sum(cond_p*np.log2(cond_p),1)
    return 2**perp



def search_deviations_1(distances:np.ndarray, perplexity=10, tolerance=0.1, iters=1000):
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
    result = np.zeros(distances.shape[0])

    max_perp = (1+tolerance) * perplexity
    min_perp = max(min_double, (1-tolerance) * perplexity) 
    
    n = distances.shape[0]
    max_deviations = 1000 * np.ones(shape=(n,))
    min_deviations = min_double*np.ones(shape=(n,)) + min_double
    
    i=0

    while i < iters and not np.array_equal(min_deviations, max_deviations):
        i+=1
        computed_deviations = (max_deviations+min_deviations)/2
        cond_p = conditional_p(distances,computed_deviations)
        perplexities = perplexity_from_conditional_p(cond_p)
        for i in range(0,n):
            if perplexities[i]>=min_perp:
                max_deviations[i] = computed_deviations[i]
            if perplexities[i]<=max_perp:
                min_deviations[i] = computed_deviations[i]
        
        max_dev_temp = max_deviations
        min_dev_temp = min_deviations
        max_deviations = np.where(perplexities>=min_perp, computed_deviations, max_dev_temp)
        min_deviations = np.where(perplexities<=max_perp, computed_deviations, min_dev_temp)
    return computed_deviations

def search_deviations_2(distances:np.ndarray, perplexity=10, tolerance=0.1, iters=1000):
    result = np.zeros(distances.shape[0])
    for i in range(distances.shape[0]):
        func = lambda sig: perplexity_from_conditional_p(conditional_p(distances[i:i+1, :], np.array([sig])))
        result[i] = __search_deviation_indiv(func, perplexity)
    return result

def __search_deviation_indiv(func, perplexity_goal, tolerance=1e-10, max_iters=1000, min_deviation=1e-20, max_deviation=10000):
    for _ in range(max_iters):
        new_deviation = (min_deviation+max_deviation)/2.
        perplexity = func(new_deviation)
        if perplexity > perplexity_goal:
            max_deviation = new_deviation
        else:
            min_deviation = new_deviation
        if np.abs(perplexity - perplexity_goal) <= tolerance:
            return new_deviation
    return new_deviation

def search_deviations_exact(distances:np.ndarray, perplexity=10, iters=1000):
    i = 0
    n = distances.shape[0]
    max_deviation = 1000 * np.ones(shape=(n,))
    min_deviation = min_double*np.ones(shape=(n,)) + min_double

    while i < iters and not np.array_equal(min_deviation, max_deviation):
        i+=1
        computed_deviation = (max_deviation+min_deviation)/2
        cond_p = conditional_p(distances,computed_deviation)
        perplexities = perplexity_from_conditional_p(cond_p)
        for i in range(0,n):
            if perplexities[i]>perplexity:
                max_deviation[i] = computed_deviation[i]
            else:
                min_deviation[i] = computed_deviation[i]
    return computed_deviation



#==========================================================================
#===========Usar distancia habiendo hecho raiz cuadrada====================
#==========================================================================
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
        result[i] = indices_sorted[i]
    
    if k is not None: #si se especifica un limite de vecinos
        return result[:,:k]
    else:
        return result

