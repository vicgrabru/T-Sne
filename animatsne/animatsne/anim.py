import numpy as np
import time
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections.abc import Sequence
from . import similarities

def _is_array_like(input) -> bool:
    return isinstance(input, (np.ndarray, Sequence)) and not isinstance(input, str)
def _assert_input(var_name:str, input=None, check="int", *,
                  accepted_values:list=None, less:int|float=None,
                  more:int|float=None, less_equal:int|float=None,
                  more_equal:int|float=None, finite=True, within_range:range=None):
    if input is not None:
        invalid_numbers = [np.nan, np.inf]
        match check:
            case "int":
                assert isinstance(input, int), "{} must be of int type".format(var_name)
            case "float":
                assert isinstance(input, float), "{} must be of float type".format(var_name)
                if finite:
                    assert input not in invalid_numbers, "{} must be finite and not NaN".format(var_name)
            case "number":
                assert isinstance(input, (int, float)), "{} must be a number".format(var_name)
                if finite:
                    assert input not in invalid_numbers, "{} must be finite and not NaN".format(var_name)
            case "str":
                assert isinstance(input, str), "{} must be of str type".format(var_name)
        if check in ["int", "float", "number"]:
            if less is not None:
                assert input<less, "{} must satisfy {}<{}".format(var_name, var_name, less)
            if more is not None:
                assert input>more, "{} must satisfy {}>{}".format(var_name, var_name, more)
            if less_equal is not None:
                assert input<=less_equal, "{} must satisfy {}<={}".format(var_name, var_name, less_equal)
            if more_equal is not None:
                assert input>=more_equal, "{} must satisfy {}>={}".format(var_name, var_name, more_equal)
            if within_range is not None:
                assert int(np.ceil(input)) in within_range, "{} must be in range(start={}, stop={}, step={})".format(var_name, within_range.start, within_range.stop, within_range.step)
        if accepted_values is not None and len(accepted_values)!=0:
            assert input in accepted_values, "only accepted values for {} are {}".format(var_name, accepted_values)

def gradient(P:np.ndarray, Q:np.ndarray, y:np.ndarray,y_dist:np.ndarray) -> np.ndarray:
    # not_diag = np.expand_dims(~np.eye(P.shape[0], dtype=bool), axis=2)
    pq = P-Q
    np.fill_diagonal(pq, 0)

    y_diff =  np.expand_dims(y,1)-np.expand_dims(y,0)
    result = np.expand_dims(pq, 2) * y_diff * np.expand_dims(1/(1+y_dist), 2)
    return 4 * result.sum(axis=1)
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
    cond = P!=0.
    return np.sum(P*np.log(P/Q, where=cond), where=cond)

class TSne():
    """Class for performing the T-Sne embedding.
    Parameters
    ----------
    n_dimensions : int, default=2
        The number of dimensions in the embedding space.
        When fitting a dataset of shape (n_samples, n_features)
        the embedding will have a shape of (n_samples, n_dimensions).

        If n_dimensions>3, only the first 3 dimensions will be displayed.
    
    init : str or Array-Like of shape (n_samples, n_dimensions), default='random'
        If Array-Like, the starting embedding of the input data.
        If str, the method for obtaining it from the input data.
        Only supported methods currently are 'pca' and 'random'.
        
        Note: to use init='pca', the animatsne must be installed with the [pca] option enabled, by running the command 'pip install animatsne[pca]'
    
    perplexity : int or float, default=30
        The perplexity represents the number of nearest neighbors
        to each sample for which t-SNE attempts to preserve distances.

        In other words, its the amount of nearest neighbors that will be kept close in the embedding.
    
    perplexity_tolerance: float, default=1e-10
        When computing P (the similarities in the input space), the accepted values
        for the perplexity will be in the range [perplexity-tol, perplexity+tol].

        Must be at least 0.
    
    metric : str, default='euclidean'
        The metric for the distance calculations.
        Currently, only supported metric is 'euclidean'
    
    early_exaggeration : int or float, default=12.
        The exaggeration factor for the first phase of the embedding.
        It affects the representation of the clusters from the input space in the embedded space.
        Specifically, it affects the size of the clusters, and the distance between them.
        
        Must be greater than 0.
    
    learning_rate : str or float, default='auto'
        The learning rate to use in the gradient descent.
        
        If float, it must be greater than 0.
        If str, it must be 'auto', in which case, the value is calculated as max(50, n_samples/early_exaggeration).
    
    starting_momentum : float, default=0.5
        Value of the momentum function in the first phase of the descent.
        It must be in the range (0., 1.)
    
    ending_momentum : float, default=0.8
        Value of the momentum function in the second phase of the descent.
        It must be in the range (0., 1.)
    
    momentum_threshold : int, default=250
        Iteration to switch to the second phase of the descent.
        In the second phase, the early exaggeration is turned off
        and the momentum function takes the value ending_momentum
    
    n_iter : int, default=1000
        Number of iterations to run, and number of frames in the animation.
        Must be at least 10
    
    iters_check : int, default=50
        The cost function will be computed every iters_check iterations.
        Must be at least 1.
    
    seed : int, default=None
        Value to seed the rng Generator through numpy.
        If None is given, the system time will be used for the seed.
    
    verbose : int, default=0
        Verbosity level (all levels include all info from previous levels).
        0 for no info, 1 for total execution time and time/iteration, 2 for evolution of the cost function

    Attributes
    ----------
    embed : None or ndarray
        Current embedding of the input data.
        If init is a string, embed remains None until fit is called.

    cost : float
        Value of the cost function for the current embedding
    
    cost_record : None or list[float]
        Record of the evolution of the cost function.
        None if fit has not been called or was called with record_cost=False
    
    embedding_record : None or list[ndarray]
        Record of the evolution of the embedding
        None if fit has not been called or was called with record_embed=False
        
    n_iter : int
        Number of iterations of t-SNE executed.
        Also the iteration of the final embedding.
    """
    def __init__(self, *,
                 n_dimensions=2,
                 init:str|Sequence|np.ndarray="random",
                 perplexity=30.,
                 perplexity_tolerance=1e-2,
                 metric='euclidean',
                 early_exaggeration=12.,
                 learning_rate:str|float="auto",
                 starting_momentum=0.5,
                 ending_momentum=0.8,
                 momentum_threshold=250,
                 n_iter=1000,
                 iters_check=50,
                 seed:int=None,
                 verbose=0,
                 ):
        #===validacion de parametros=================================================================================
        self.__init_validation(n_dimensions, perplexity, perplexity_tolerance, metric, init, early_exaggeration, learning_rate, n_iter, starting_momentum, ending_momentum, momentum_threshold, seed, verbose, iters_check)

        #===inicializacion de la clase===============================================================================
        self._n_dimensions = n_dimensions if isinstance(n_dimensions, int) else int(np.floor(n_dimensions))
        self._perplexity = perplexity
        self._perplexity_tolerance = perplexity_tolerance
        self._metric = metric.lower()
        if isinstance(init, Sequence) and not isinstance(init, str):
            self._init = np.array(init)
        else:
            self._init = init
        self._learning_rate = learning_rate
        self.__lr = None
        self.n_iter = n_iter
        self._momentum_start = starting_momentum
        self._momentum_end = ending_momentum
        self._momentum_threshold = momentum_threshold
        self._early_exaggeration = 12. if early_exaggeration is None else early_exaggeration
        self._verbose = verbose
        self._iters_check = iters_check
        self._seed = int(time.time()) if seed is None else seed
        self._rng = np.random.default_rng(self._seed)

        
        #=== Plotting Params
        self._plotting_fig, self._plotting_ax = plt.subplots() if self._n_dimensions==2 else plt.subplots(subplot_kw=dict({"projection": "3d"}))
        self._plotting_labels = None
        self._plotting_colors = None
        self._plotting_markers = None
        self._plotting_size = 5
        
        
        #=== Auxiliary Params ====================================================================================
        self._best_iter = None
        self._best_cost = None
        self.cost = None
        self._init_embed = None
        self.embed = None
        self._update = None
        self.embedding_record = None
        self.cost_record = None

    def __init_validation(self,
                          n_dimensions,
                          perplexity,
                          perplexity_tolerance,
                          metric,
                          init,
                          early_exaggeration,
                          learning_rate,
                          n_iter,
                          starting_momentum,
                          ending_momentum,
                          momentum_threshold,
                          seed,
                          verbose,
                          iters_check):

        # N dimensions: int
        _assert_input("n_dimensions", n_dimensions, "int", more=1)
        if n_dimensions is not None and n_dimensions>3:
            print("**Warning: Only the first 3 components will be displayed**")
        
        # Perplexity: int|float
        _assert_input("perplexity", perplexity, "number", more=0.)
        
        # Perplexity tolerance: float
        _assert_input("perplexity_tolerance", perplexity_tolerance, "float", more_equal=0.)
        
        # Metric: str
        _assert_input("metric", metric, "str", accepted_values=["euclidean", "precomputed"])
        
        # Init: numpy.ndarray
        if init is not None:
            if isinstance(init, str):
                assert init.lower() in ["random", "pca"], "Only init_method values accepted are 'random', 'precomputed' and 'pca'"
                if metric is not None and init.lower()=="pca":
                    assert metric.lower()!="precomputed", "Init cannot be 'pca' when metric is 'precomputed'"
                if init.lower=="pca":
                    import importlib.util as imp_ut
                    spec = imp_ut.find_spec("scikit-learn")
                    assert spec is not None, "If you want to use pca initialization, install the scikit-learn package, or install animatsne with the [pca] option"
            elif _is_array_like(init):
                init = np.array(init)
                
                assert isinstance(init.dtype, np.number), "Data type of the initial embedding must be a number"
                assert np.all(np.isfinite(init)), "The initial embedding must not contain NaN or an infinite number"
                if n_dimensions is not None:
                    assert init.shape[1]==n_dimensions, "The initial embedding must have the number of dimensions provided"
            else:
                raise ValueError("init must be a str or array-like")
        
        # Early exaggeration: floatless:int|float=None
        _assert_input("early_exaggeration", early_exaggeration, "number", more=0.)
        
        # Learning rate: float|str
        if learning_rate is not None: # learning_rate: float
            assert isinstance(learning_rate, (str, int, float)), "learning_rate must a number or 'auto'"
            if isinstance(learning_rate, str):
                assert learning_rate == "auto", "The only str value acceptable for learning_rate is 'auto'"
            elif isinstance(learning_rate, (int, float)):
                _assert_input("learning_rate", learning_rate, "number", more=0.)
        
        # Max iter: int
        _assert_input("n_iter", n_iter, "int", more_equal=10)
        
        # Starting Momentum: float
        _assert_input("starting_momentum", starting_momentum, "float", more=0., less=1.)
        
        # Ending Momentum: float
        _assert_input("ending_momentum", ending_momentum, "float", more=0., less=1.)
        
        # Momentum threshold: float
        _assert_input("momentum_threshold", momentum_threshold, "int", within_range=range(n_iter))
        
        # seed: int
        _assert_input("seed", seed, "int", more_equal=0)
        
        # Iters check: int
        _assert_input("iters_check", iters_check, "int", more_equal=1)
        if iters_check is not None:
            assert iters_check<=n_iter, "iters_check cannot be greater than n_iter"
        
        # Verbose: int
        _assert_input("verbose", verbose, "int", more_equal=0)
    def __input_validation(self, input, labels=None):
        assert _is_array_like(input), "The given input is not array-like"
        result = np.array(input)
        
        if self._metric=="precomputed":
            assert result.ndim==2 and result.shape[0]==result.shape[1], "When metric is 'precomputed', input data must be a square distance matrix"
        
        assert len(result)>=2*int(self._perplexity), "The number of samples cannot be lower than twice the given Perplexity"

        if self._init is not None and _is_array_like(self._init) and len(input) != self._init.shape[0]:
            raise ValueError("The input data must have the same number of samples as the given embedding")
        if labels is not None:
            assert _is_array_like(labels),"labels is not array-like"
            self._plotting_labels = np.array(labels, dtype=np.str_)
            assert self._plotting_labels.ndim==1, "labels must be a 1D array"
            self._plotting_markers = np.empty_like(self._plotting_labels)
            unicos = np.unique(self._plotting_labels)
            
            marker_options = ['o','*','v','^','<','>']
            if np.all(np.char.isnumeric(self._plotting_labels)):
                self._plotting_colors = np.array(labels, dtype=int)
                if unicos.shape[0]>5:
                    for i in range(unicos.shape[0]):
                        indices = np.argwhere(self._plotting_labels==unicos[i])
                        self._plotting_markers[indices] = marker_options[int(i%len(marker_options))]
                else:
                    self._plotting_markers.fill('o')
                
            else:
                self._plotting_colors = np.empty_like(labels, dtype=int)
                n_unique = unicos.shape[0]
                
                if n_unique<=5:
                    for i in range(n_unique):
                        indices = np.argwhere(self._plotting_labels==unicos[i])
                        self._plotting_colors[indices] = i
                    self._plotting_markers.fill('o')
                else:
                    n = max(np.floor(np.sqrt(n_unique)), 5)
                    colores = np.array(range(n))
                    for i in range(n_unique):
                        indices = np.argwhere(self._plotting_labels==unicos[i])
                        self._plotting_colors[indices] = colores[np.divmod(i, n)[0]]
                        self._plotting_markers[indices] = marker_options[int(i%len(marker_options))]
        else:
            self._plotting_markers = np.full(shape=len(result), fill_value='o')
            self._plotting_colors = np.full(shape=len(result), fill_value=1)
        if result.ndim>2:
            return result.reshape((len(result), np.prod(result.shape[1:])))
        return result
    def __rand_embed(self, input, n_dimensions) -> np.ndarray:
        assert n_dimensions is not None
        n_samples = len(input)
        if self._init is None:
            return self._rng.standard_normal(size=(n_samples, n_dimensions))
        elif isinstance(self._init, str):
            if self._init.lower()=="pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_dimensions, svd_solver="randomized", random_state=self._seed)
                pca.set_output(transform="default")
                data_embedded = pca.fit_transform(input).astype(np.float32, copy=False)
                return data_embedded / np.std(data_embedded[:, 0]) * 1e-4
            else: #init = "random"
                return self._rng.standard_normal(size=(n_samples, n_dimensions))
        else:
            return self._init.copy()
    
    def fit(self, input, labels=None, record_embed=False, record_cost=False, gif_filename=None, gif_kwargs=None) -> np.ndarray:
        """Fit the given data and display the embedding process
    
        Parameters
        ----------
        input: array-like of shape (n_samples, n_features).
            The data to fit.
        
        labels: None or array-like of shape (n_samples,).
            Array with the labels to assign each sample in the animation.
            If None, all samples will be asumed to have the same label.
        
        record_embed : boolean, default=False.
            If True, a record of each iteration of embedding is kept in a list.
            This list is stored in the parameter embedding_record.
        
        record_cost : boolean, default=False.
            If True, a record of the value of the cost function throughout the embedding process is kept in a dictionary.
            This dictionary is stored in the parameter cost_record.
        
        gif_filename: str or None. default=None. Optional.
            The file to output the animation to.
            If None, the animation is displayed once
        
        gif_kwargs: dict
            Additional keyword arguments for the gif save method.
        """

        #====Tiempo de inicio para verbosidad====================================================================================================================
        t0 = time.time_ns()

        #====Input con dimensiones correctas=====================================================================================================================
        X = self.__input_validation(input, labels)

        self._init_embed = self.__rand_embed(X, self._n_dimensions)
        
        #====Ajuste del learning rate============================================================================================================================
        if self._learning_rate == "auto":
            self.__lr = len(X) / self._early_exaggeration
            self.__lr = np.maximum(self.__lr, 50)
        else:
            self.__lr = self._learning_rate

        #====Obtener P===========================================================================================================================================
        if self._metric=="precomputed":
            p = similarities.joint_probabilities_gaussian(X, self._perplexity, self._perplexity_tolerance)
        else:
            dist_original = similarities.pairwise_euclidean_distance(X)
            p = similarities.joint_probabilities_gaussian(dist_original, self._perplexity, self._perplexity_tolerance)
            del dist_original
        
        #===Coste inicial
        dist_embed = similarities.pairwise_euclidean_distance(self._init_embed)
        q = similarities.joint_probabilities_student(dist_embed)
        c = kl_divergence(p, q)
        self.cost = c
        self._best_cost = c
        self._best_iter = 0
        del q, dist_embed

        #====Descenso de gradiente===============================================================================================================================
        
        self._update = np.zeros_like(self._init_embed)
        self.embed = self._init_embed.copy()

        if record_embed:
            self.embedding_record = [self.embed.copy()]
        if record_cost:
            self.cost_record = {0: self.cost}
        
        if self._n_dimensions==2:
            line = self._plotting_ax.scatter(self._init_embed.T[0], self._init_embed.T[1], label=self._plotting_labels, c=self._plotting_colors, s=self._plotting_size)
        else:
            line = self._plotting_ax.scatter(self._init_embed.T[0], self._init_embed.T[1], self._init_embed.T[2], label=self._plotting_labels, c=self._plotting_colors, s=self._plotting_size)
        
        if self._plotting_labels is not None:
            leg = self._plotting_ax.legend(*line.legend_elements(), loc="lower right")
            self._plotting_ax.add_artist(leg)
        plt.title("Initial embedding")
        ani = animation.FuncAnimation(self._plotting_fig, self.__update_anim, self.n_iter, fargs=[p, self._plotting_ax], interval=100, repeat=False)

        if gif_filename is None:
            plt.show()
        else:
            ani.save(gif_filename, **gif_kwargs)
        
        
        #====Salida por consola de verbosidad====================================================================================================================
        if self._verbose>0:
            t = (time.time_ns()-t0)*1e-9
            tiempos = np.array([t, t/self.n_iter])
            tiempos_exacto = np.floor(tiempos)
            tH = np.floor(tiempos_exacto/3600, dtype=int)
            tM = np.floor(tiempos_exacto/60, dtype=int)-60*tH
            tS = tiempos - np.array(tH*3600+tM*60, dtype=float)
            
            strings = []
            for i in range(len(tiempos)):
                if tiempos[i]>3600:
                    strings.append("(h:min:sec): {}:{}:{}".format(tH[i],tM[i],tS[i]))
                elif tiempos[i]>60:
                    strings.append("(min:sec): {}:{}".format(tM[i],tS[i]))
                else:
                    strings.append("(s): {}".format(tS[i]))
            
            print("====================================")
            print("Embedding process finished")
            print("Execution time " + strings[0])
            print("Time/Iteration " + strings[1])
            print("====================================")
            del t0,t,tiempos,tiempos_exacto,tS,tM,tH,strings
        
        return self.embed

    def __update_embed(self, i, affinities):
        embed_dist = similarities.pairwise_euclidean_distance(self.embed)
        q = similarities.joint_probabilities_student(embed_dist)

        # Momentum switch
        if i<self._momentum_threshold:
            p = self._early_exaggeration*affinities
            momentum = self._momentum_start
        else:
            p = affinities
            momentum = self._momentum_end
        
        # Cost
        if i%self._iters_check==0:
            self.cost = kl_divergence(p, q)
            if self._best_cost is None or self.cost<self._best_cost:
                self._best_iter = i
                self._best_cost = self.cost
            if self.cost_record is not None:
                self.cost_record.update({i: self.cost})
            if self._verbose>1:
                print("Cost(i={}): {:.5f}".format(i, self.cost))

        # Calculo de nuevo embed
        grad = gradient(p, q, self.embed, embed_dist)
        self._update = momentum*self._update - grad*self.__lr
        self.embed += self._update
        if self.embedding_record is not None:
            self.embedding_record.append(self.embed.copy())
    def __update_anim(self, i, affinities, ax:Axes):
        self.__update_embed(i, affinities)
        
        #===Plotting===============================================================================
        ax.clear()
        
        x = self.embed.T[0]
        y = self.embed.T[1]
        if self._n_dimensions==3:
            z = self.embed.T[2]
        
        leg_aux1 = []
        leg_aux2 = []

        for m in np.unique(self._plotting_markers):
            cond = self._plotting_markers==m
            l = None if self._plotting_labels is None else self._plotting_labels[cond]
            if self._n_dimensions==2:
                line = ax.scatter(x[cond], y[cond], marker=m, c=self._plotting_colors[cond], label=l, s=self._plotting_size)
            else:
                line = ax.scatter(x[cond], y[cond], z[cond], marker=m, c=self._plotting_colors[cond], label=l, s=self._plotting_size)
            
            # PARA LA LEYENDA
            aux1, aux2 = line.legend_elements()
            leg_aux1.extend(aux1)
            leg_aux2.extend(aux2)

        # Labels
        if self._plotting_labels is not None:
            leg = ax.legend(leg_aux1, leg_aux2, loc="lower right")
            ax.add_artist(leg)

        # Title
        plt.title("Current Iteration: {}/{} \n Best cost: i={}, cost={:.3f}".format(i+1, self.n_iter, self._best_iter, self._best_cost))

    def get_best_embed_info(self):
        """Returns the best cost achieved and the iteration it belongs to
        """
        return self._best_cost, self._best_iter