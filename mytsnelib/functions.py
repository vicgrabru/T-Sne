import numpy as np
import matplotlib.pyplot as plt
from mytsnelib import similarities
import time


#===Gradiente====================================================================================================
def gradient(P, Q, y, y_dist, caso="safe") -> np.ndarray:
    match caso:
        case "safe":
            return __gradient_safe(P,Q,y,y_dist)
        case "forces":
            return __gradient_forces(P,Q,y,y_dist)
        case "forces_v2":
            return __gradient_forces_v2(P,Q,y,y_dist)
        case _:
            raise ValueError("Only accepted cases are safe, forces, and forces_v2")
#----------------------------------------------------------------------------------------------------------------
def __gradient_safe(P, Q, y, y_dist) -> np.ndarray:
    pq_diff = P - Q
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0)

    aux = (1+np.copy(y_dist))**(-1)
    return 4 * np.sum(np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux,2), axis=1)
def __gradient_forces(P, Q, y, y_dist) -> np.ndarray:
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0)

    # paso 1: obtener Z
    dists = (1+np.copy(y_dist))**(-1)
    np.fill_diagonal(dists, 0.)
    z = dists.sum()

    # paso 2: calcular fuerzas atractivas y repulsivas
    # fuerzas atractivas
    pq = P*Q*z
    np.fill_diagonal(pq, 0.)
    attractive = np.expand_dims(pq, 2)* y_diff

    # fuerzas repulsivas
    q2 = (Q**2)*z
    np.fill_diagonal(q2, 0.)
    repulsive = np.expand_dims(q2, 2)* y_diff

    # paso 3: combinacion
    return 4*(np.sum(attractive, 1) - np.sum(repulsive, 1))
def __gradient_forces_v2(P, Q, y, y_dist) -> np.ndarray:
    """
    Optimizaciones sacadas de esta pagina
    Optimizacion 1 -> https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html#attractive-forces
    Optimizacion 2 -> https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html#repulsive-forces
    """
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0)

    # paso 1: obtener Z
    dists = 1/(1+y_dist)
    np.fill_diagonal(dists, 0.)
    z = dists.sum()

    # paso 2: calcular fuerzas atractivas y repulsivas
    
    # fuerzas atractivas
    # Optimizacion 1: considerar solo los 3*Perplexity vecinos mas cercanos
    
    pq = P*Q*z
    np.fill_diagonal(pq, 0.)
    attractive = np.multiply(np.expand_dims(pq, 2), y_diff)

    # fuerzas repulsivas
    # Optimizacion 2: TODO
    q2 = (Q**2)*z
    np.fill_diagonal(q2, 0.)
    repulsive = np.multiply(np.expand_dims(q2, 2), y_diff)

    # paso 3: combinacion
    return 4*(np.sum(attractive, 1) - np.sum(repulsive, 1))

#===Funcion de coste: Kullback-Leibler divergence================================================================
def kl_divergence(P, Q) -> float:
    """Computes the Kullback-Leibler divergence
    Parameters
    ----------
        high_dimension_p: ndarray of shape (n_samples, n_samples)
            The joint probabilities for the samples in the original dimension.

        low_dimension_p: ndarray of shape (n_samples, n_samples)
            The joint probabilities for the samples embedded in the lower dimension.

    Returns
    -------
        divergence : double.
            The divergence.
    """
    
    
    low = np.copy(Q)

    low[low<=0] = np.finfo(Q.dtype).eps
    
    aux = np.divide(P, low)
    del(low)

    aux[aux==0] = np.finfo(Q.dtype).eps

    result = np.multiply(P, np.log(aux))
    del(aux)

    return np.sum(result)

class TSne():
    """Class for performing the T-Sne embedding.
    Parameters
    ----------
    n_dimensions : int, default=2
        The number of dimensions in the embedding space.
    perplexity : float, default=30.0
        TODO: explicarlo
    perplexity_tolerance: float, default=1e-10
        TODO: explicarlo
    metric : str, default='euclidean'
        The metric for the distance calculations
    init_method : str, default="random"
        TODO: explicarlo
    init_embed : ndarray of shape (n_samples, n_components), default=None
        TODO: explicarlo
    early_exaggeration : float, default=None
        TODO: explicarlo
    learning_rate : float, default=500.
        TODO: explicarlo
    max_iter : int, default=1000
        TODO: explicarlo
    momentum_params : array-like of shape (3,), default=[250.,0.5,0.8]
        TODO: explicarlo
    seed : int, default=None
        TODO: explicarlo
    verbose : int, default=0
        Verbosity level (all levels include all info from previous levels):
            0 displays no info
            1 displays total execution time and time / iteration
            2 displays
        TODO: explicarlo
    iters_check : int, default=50
        TODO: explicarlo

    Attributes
    ----------
    embed_history

    
    """


    def __init__(self, *, n_dimensions=2, perplexity=30., perplexity_tolerance=1e-10,
                 metric='euclidean', init_method="random", init_embed=None, early_exaggeration=12.,
                 learning_rate:str|float="auto", max_iter=1000, momentum_params=[250.,0.5,0.8], seed:int=None, verbose=0, iters_check=50):
        
        #===validacion de parametros=================================================================================
        self.__input_validation(n_dimensions, perplexity, perplexity_tolerance, metric, init_method, init_embed,early_exaggeration, learning_rate, max_iter, momentum_params, seed, verbose, iters_check)

        #===inicializacion de la clase===============================================================================
        self.n_dimensions = n_dimensions
        self.perplexity = perplexity
        self.perplexity_tolerance = perplexity_tolerance
        self.metric = metric
        self.init_method = init_method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_embed = init_embed
        self.momentum_params = momentum_params
        self.early_exaggeration = 12. if early_exaggeration is None else early_exaggeration
        # self.n_neighbors = 3*perplexity + 1
        self.verbose = verbose
        self.iters_check = iters_check
        self.rng = np.random.default_rng(int(time.time())) if seed is None else np.random.default_rng(seed)
        # self.random_state = np.random.RandomState(int(time.time())) if seed is None else np.random.RandomState(seed)

        #===parametros auxiliares====================================================================================
        self.current_embed = None
        # self.embed_history = []
        self.cost_history = []
        self.best_cost_iter = None
        self.best_cost_embed = None
        # self.cost_history_embed = []
        # self.best_iter_cost = None
        self.fitting_done = False
        self.current_embed_dist = None
        
        #===parametros para las metricas de rendimiento==============================================================
        self.t_diff_distancias_og = None
        self.t_diff_p = None
        self.t_diff_dist_embed = []
        self.t_diff_q = []
        self.t_diff_grad = []
    def __input_validation(self,n_dimensions,perplexity,perplexity_tolerance,metric,init_method,init_embed,
                           early_exaggeration,learning_rate,max_iter,momentum_params, seed, verbose, iters_check):
        accepted_methods = ["random", "precomputed"]
        accepted_metrics = ["euclidean"]
        accepted_momentum_param_types = [np.float64,np.float32]
        invalid_numbers = np.array([np.nan, np.inf])

        if n_dimensions is not None: # n_dimensions: int
            if not isinstance(n_dimensions, int):
                raise ValueError("n_dimensions must be of int type")
            elif n_dimensions in invalid_numbers:
                raise ValueError("n_dimensions must be finite and not NaN")
            elif n_dimensions<1:
                raise ValueError("n_dimensions must be a positive number")
            elif n_dimensions>3:
                print("**Warning: If you use more than 3 dimensions, you will not be able to display the embedding**")
        if perplexity is not None: # perplexity: float
            if not isinstance(perplexity, float):
                raise ValueError("perplexity must be of float type")
            elif perplexity in invalid_numbers:
                raise ValueError("perplexity cannot be infinite or NaN")
            elif perplexity <= 0.:
                raise ValueError("perplexity must be a positive number")
        if perplexity_tolerance is not None: # perplexity_tolerance: float
            if not isinstance(perplexity_tolerance, float):
                raise ValueError("perplexity_tolerance must be of float type")
            elif perplexity_tolerance in invalid_numbers:
                raise ValueError("perplexity_tolerance must be finite and not NaN")
            elif perplexity_tolerance < 0:
                raise ValueError("perplexity_tolerance must be a positive number or 0")
        if metric is not None: # metric: str
            if not isinstance(metric, str):
                raise ValueError("metric must be of str type")
            elif metric not in accepted_metrics:
                raise ValueError("Only currently accepted metric is euclidean")
        if init_method is not None: # init_method: str
            if not isinstance(init_method, str):
                raise ValueError("init_method must be of str type")
            else: 
                if init_method not in accepted_methods:
                    raise ValueError("Only init_method values accepted are random and precomputed")
        if init_embed is not None: # init_embed: ndarray of shape (n_samples, n_features)
            if isinstance(init_embed, np.ndarray):
                if not isinstance(init_embed.ndtype, np.number):
                    raise ValueError("Data type of the initial embedding must be a number")
                elif len(np.intersect1d(invalid_numbers, np.reshape(init_embed, [1,-1])))>0:
                    raise ValueError("init_embed cant have NaN or an infinite number")
            else:
                raise ValueError("init_embed must be a ndarray")
        if early_exaggeration is not None: # early_exaggeration: float
            if not isinstance(early_exaggeration, float):
                raise ValueError("early_exaggeration must be of float type")
            elif early_exaggeration in invalid_numbers:
                raise ValueError("early_exaggeration must be finite and not NaN")
            elif early_exaggeration <=0.:
                raise ValueError("early_exaggeration must be a positive number")
        if learning_rate is not None: # learning_rate: float
            if isinstance(learning_rate, str):
                if learning_rate != "auto":
                    raise ValueError("The only str value acceptable for learning_rate is 'auto'")
            elif isinstance(learning_rate, float):
                if learning_rate in invalid_numbers:
                    raise ValueError("learning_rate must be finite and not NaN")
                elif learning_rate <=0.:
                    raise ValueError("learning_rate must be a positive number")
            else:
                raise ValueError("learning_rate must be of float type")
            
        if max_iter is not None: # max_iter: int
            if not isinstance(max_iter, int):
                raise ValueError("max_iter must be of int type")
            elif max_iter in invalid_numbers:
                raise ValueError("max_iter must be finite and not NaN")
            elif max_iter <1:
                raise ValueError("max_iter must be a positive number")
        if momentum_params is not None: # momentum_params: ndarray of shape (3,)
            if not isinstance(momentum_params, np.ndarray):
                if np.array(momentum_params).shape!=(3,):
                    raise ValueError("momentum_params must be a ndarray of shape (3,)")
            elif momentum_params.shape!=(3,):
                raise ValueError("momentum_params must be a ndarray of shape (3,)")
            elif momentum_params.dtype not in accepted_momentum_param_types:
                raise ValueError("The elements of momentum_params must be float(at least float32)")
            elif len(np.intersect1d(momentum_params, invalid_numbers))>0:
                raise ValueError("init_embed cant have NaN or an infinite number")
            elif np.min(momentum_params)<=0.:
                raise ValueError("All elements must be positive numbers")
            elif not (momentum_params[0]).is_integer():
                raise ValueError("The time threshold must be a whole number")
        if seed is not None: # seed: int
            if not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            elif seed<0:
                raise ValueError("seed must be a positive integer")
        if verbose is not None: # verbose: int
            if not isinstance(verbose, int):
                raise ValueError("verbose must be an integer")
            elif verbose<0:
                raise ValueError("verbose must be a positive number")
        if iters_check is not None: #iters_check: int
            if not isinstance(iters_check, int):
                raise ValueError("iters_check must be an integer")
            elif iters_check<1:
                raise ValueError("iters_check must be at least 1")
            elif iters_check>max_iter:
                raise ValueError("iters_check cannot be greater than max_iter")
    def __fit_input_validation(self, input, *, embed:np.ndarray=None):
        if isinstance(input, np.ndarray):
            result = input
        elif isinstance(input, (tuple, list)):
            result = np.array(input)
        else:
            raise ValueError("The given input is not array-like")
        
        if len(result)<10:
            raise ValueError("Not enough samples. Must be at least 10 samples.")
        if embed is not None:
            if len(input) != len(embed):
                raise ValueError("The input data must have the same number of samples as the given embedding")
        if len(result) <= 3*self.perplexity + 1:
            raise ValueError("The number of samples cannot be lower than the number of neighbors")
        
        if result.ndim>2:
            dims = 1
            for d in result.shape[1:]:
                dims*=d
            return result.reshape((len(result), dims))
        return result
    def __rand_embed(self, *, n_samples:int, n_dimensions:int) -> np.ndarray:
        """Returns a random embedding following the given parameters.

        Parameters
        ----------
        n_samples: int.
            The number of samples to have in the embedding.
        n_dimensions: int.
            The number of dimentsions in the embedding.
        
        Returns
        ----------
        embed: ndarray of shape (n_samples, n_features).
            Random embedding that follows a standard distribution.

        """
        assert n_samples is not None
        assert n_dimensions is not None
        result = self.rng.standard_normal(size=(n_samples, n_dimensions))
        # result = self.random_state.standard_normal(size=(n_samples, n_dimensions))
        return result

    def fit(self, input, compute_cost=True) -> np.ndarray:
        """Fit the given data and perform the embedding

        Parameters
        ----------
        X: array of shape (n_samples, n_features).
            The data to fit.
        classes: 1-D array of size (n_samples). Optional.
            Array with the class that each element in X belongs to.
        """

        #====Input con dimensiones correctas y embed inicial=====================================================================================================
        X = self.__fit_input_validation(input, embed=self.init_embed)
        if self.init_embed is None:
            self.init_embed = self.__rand_embed(n_samples=len(X), n_dimensions=self.n_dimensions)

        #====Ajuste del learning rate============================================================================================================================
        if isinstance(self.learning_rate, str):
            self.learning_rate = np.floor(len(X)/self.early_exaggeration)
        else:
            self.learning_rate = max(self.learning_rate, np.floor(len(X)/self.early_exaggeration))

        #====Tiempo de inicio para verbosidad====================================================================================================================
        t0 = time.time_ns()
        
        #====Obtener P===========================================================================================================================================
        dist_original = similarities.pairwise_euclidean_distance(X) #solo es necesario para calcular P
        p = similarities.joint_probabilities_gaussian(dist_original, self.perplexity, self.perplexity_tolerance)
        del(dist_original) #dist_original ya no hace falta

        #====Descenso de gradiente===============================================================================================================================
        self.__gradient_descent(self.max_iter, p, compute_cost)
        del(p)
        #====Salida por consola de verbosidad====================================================================================================================
        if self.verbose>0:
                print("=================================================================")
                print("Embedding process finished")
                t_diff = (time.time_ns()-t0)*1e-9
                t_diff_clean = time.gmtime(t_diff)
                t_diff_exact_ms = t_diff - t_diff_clean.tm_sec
                print("Execution time (min:sec): {}.{}".format(time.strftime("%M:%S", t_diff_clean), str(t_diff_exact_ms)[2:8]))
                t_iter = t_diff/self.max_iter
                del(t_diff, t_diff_clean, t_diff_exact_ms)
                t_iter_clean = time.gmtime(t_iter)
                t_iter_exact_ms = t_iter - t_iter_clean.tm_sec
                print("Time/Iteration (s): {}.{}".format(time.strftime("%S", t_iter_clean), str(t_iter_exact_ms)[2:8]))
                print("=================================================================")
                del(t_iter, t_iter_clean, t_iter_exact_ms)
        del(t0)

        #====Establecer indicador de que se ha hecho el embedding================================================================================================
        self.fitting_done = True

        return np.array(self.current_embed)
    def __gradient_descent(self, t, p, compute_cost=True):
        self.current_embed = np.copy(self.init_embed)
        
        #====dist_embed==========================================================================================================================================
        self.current_embed_dist = similarities.pairwise_euclidean_distance(self.current_embed)
        
        #====q===================================================================================================================================================
        q = similarities.joint_probabilities_student(self.current_embed_dist)
        
        
        # self.embed_history.append(self.current_embed); self.embed_history.append(self.current_embed)
        # self.embed_dist_history.append(self.prev_embed_dist); self.embed_dist_history.append(self.prev_embed_dist)
        # self.q_history.append(q); self.q_history.append(q)

        

        if compute_cost:
            cost = kl_divergence(p, q)
            self.best_cost_iter = 1
            self.best_cost_embed = np.copy(self.current_embed)
            self.cost_history.append(cost); self.cost_history.append(cost)
            # self.cost_history_embed.append(self.current_embed); self.cost_history_embed.append(self.current_embed)

        lr = self.learning_rate
        # early = self.early_exaggeration
        momentum = self.momentum_params[1]
        previous_embed = np.copy(self.current_embed)
        for i in range(2,t):
            #====grad================================================================================================================================================
            # grad = gradient(p, q*early, self.current_embed, self.current_embed_dist, caso="forces")
            grad = gradient(p, q, self.current_embed, self.current_embed_dist, caso="forces")

            #====embed===============================================================================================================================================
            #y{i} = y{i-1} + learning_rate*gradiente + momentum(t) * (y{i-1} - y{i-2})
            # y = self.embed_history[-1] - lr*grad + momentum*(self.embed_history[-1]-self.embed_history[-2])
            new_y = self.current_embed - lr*grad + momentum*(self.current_embed-previous_embed)

            #====momentum change=====================================================================================================================================
            if i==self.momentum_params[0]:
                # early = 1
                momentum = self.momentum_params[2]

            #====dist_embed==========================================================================================================================================
            self.current_embed_dist = similarities.pairwise_euclidean_distance(new_y)
            #====q===================================================================================================================================================
            q = similarities.joint_probabilities_student(self.current_embed_dist)
            
            
            
            # self.embed_history.append(new_y)
            # self.embed_dist_history.append(self.prev_embed_dist)
            # self.q_history.append(q)

            if compute_cost:
                if i%self.iters_check==0 or i==t-1:
                    cost = kl_divergence(p, q)
                    if cost<np.min(self.cost_history):
                        self.best_cost_iter = i
                        self.best_cost_embed = np.copy(new_y)
                    self.cost_history.append(cost)
                    # self.cost_history_embed.append(y)
            
            #actualizar y(t-1), y(t-2)
            previous_embed = np.copy(self.current_embed)
            self.current_embed = np.copy(new_y)

    def get_best_embedding_cost(self) -> np.ndarray:
        assert self.fitting_done
        return self.best_cost_embed

    

    