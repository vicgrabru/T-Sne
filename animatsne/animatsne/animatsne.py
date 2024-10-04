import numpy as np
import time
import gc
from matplotlib.markers import MarkerStyle
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.artist import Artist
from matplotlib.path import Path
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections.abc import Sequence
from animatsne import similarities

def _is_array_like(input) -> bool:
    return isinstance(input, (np.ndarray, Sequence)) and not isinstance(input, str)

#===Gradiente====================================================================================================
def gradient(P:np.ndarray, Q:np.ndarray, y:np.ndarray,y_dist:np.ndarray) -> np.ndarray:
    # not_diag = np.expand_dims(~np.eye(P.shape[0], dtype=bool), axis=2)
    pq = P-Q
    np.fill_diagonal(pq, 0)

    y_diff =  np.expand_dims(y,1)-np.expand_dims(y,0)
    result = np.expand_dims(pq, 2) * y_diff * np.expand_dims(1/(1+y_dist), 2)
    return 4 * result.sum(axis=1)
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
    cond = P!=0.
    return np.sum(P*np.log(P/Q, where=cond), where=cond)

class TSne():
    """Class for performing the T-Sne embedding.
    Parameters
    ----------
    n_dimensions : int, default=2
        The number of dimensions in the embedding space.
    perplexity : int or float, default=30.0
        TODO: explicarlo
    perplexity_tolerance: float, default=1e-10
        TODO: explicarlo
    metric : str, default='euclidean'
        The metric for the distance calculations
    init : str or Array-Like of at least 2 dimensions, default='random'
        TODO: explicarlo
    early_exaggeration : int or float, default=12.
        TODO: explicarlo
    learning_rate : str or float, default='auto'
        TODO: explicarlo
    n_iter : int, default=1000
        TODO: explicarlo
    momentum_params : array-like of shape (3,), default=[250.,0.5,0.8]
        TODO: explicarlo
    seed : int, default=None
        TODO: explicarlo
    verbose : int, default=0
        Verbosity level (all levels include all info from previous levels):
            0 displays no info
            1 displays total execution time and time / iteration
            2 displays iteration info every *iters_check* iterations
        TODO: explicarlo
    iters_check : int, default=50
        TODO: explicarlo

    Attributes
    ----------

    
    """
    def __init__(self, *, n_components=2, perplexity=30., perplexity_tolerance=1e-2,
                 metric='euclidean', init:str|Sequence|np.ndarray="random", early_exaggeration=12.,
                 learning_rate:str|float="auto", n_iter=1000, starting_momentum=0.5, ending_momentum=0.8, momentum_threshold=250, seed:int=None, verbose=0, iters_check=50):
        
        #===validacion de parametros=================================================================================
        self.__init_validation(n_components, perplexity, perplexity_tolerance, metric, init, early_exaggeration, learning_rate, n_iter, starting_momentum, ending_momentum, momentum_threshold, seed, verbose, iters_check)

        #===inicializacion de la clase===============================================================================
        self.n_components = n_components if isinstance(n_components, int) else int(np.floor(n_components))
        self.perplexity = perplexity
        self.perplexity_tolerance = perplexity_tolerance
        self.metric = metric.lower()
        if isinstance(init, Sequence) and not isinstance(init, str):
            self.init = np.array(init)
        else:
            self.init = init
        self.learning_rate = learning_rate
        self.lr = None
        self.n_iter = n_iter
        self.momentum_start = starting_momentum
        self.momentum_end = ending_momentum
        self.momentum_threshold = momentum_threshold
        self.early_exaggeration = 12. if early_exaggeration is None else early_exaggeration
        self.verbose = verbose
        self.iters_check = iters_check
        self.seed = int(time.time()) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.n_neighbors = int(perplexity)

        
        #===plotting de la representacion
        
        # self.plotting_fig, self.plotting_ax = plt.subplots(layout='constrained') if self.n_components==2 else plt.subplots(layout='constrained',subplot_kw=dict({"projection": "3d"}))
        self.plotting_fig, self.plotting_ax = plt.subplots() if self.n_components==2 else plt.subplots(subplot_kw=dict({"projection": "3d"}))
        self.plotting_labels = None
        self.plotting_colors = None
        self.plotting_markers = None
        self.plotting_marker_options = ['o','*','v','^','<','>']
        self.plotting_marker_result = None
        self.plotting_size = 5
        
        
        #===parametros auxiliares====================================================================================
        self.embedding_finished = False
        self.embed_i = None
        self.embed_cost = None
        self.init_embed = None
        self.current_embed = None
        self.best_embed = None
        self.update = None

    def __init_validation(self,n_components,perplexity,perplexity_tolerance,metric,init,
                           early_exaggeration,learning_rate,n_iter, starting_momentum, ending_momentum, momentum_threshold, seed, verbose, iters_check):
        accepted_inits = ["random", "pca"]
        accepted_metrics = ["euclidean", "precomputed"]
        invalid_numbers = [np.nan, np.inf]

        # N dimensions
        if n_components is not None: # n_dimensions: int
            if not isinstance(n_components, (int, float)):
                raise ValueError("n_dimensions must be of int type")
            elif n_components in invalid_numbers:
                raise ValueError("n_dimensions must be finite and not NaN")
            elif n_components<1:
                raise ValueError("n_dimensions must be a positive number")
            elif n_components>3:
                print("**Warning: If you use more than 3 dimensions, you will not be able to display the embedding**")
        
        # Perplexity
        if perplexity is not None: # perplexity: int or float
            if not isinstance(perplexity, (int,float)):
                raise ValueError("perplexity must a number")
            elif perplexity in invalid_numbers:
                raise ValueError("perplexity cannot be infinite or NaN")
            elif perplexity <= 0.:
                raise ValueError("perplexity must be a positive number")
        
        # Perplexity tolerance
        if perplexity_tolerance is not None: # perplexity_tolerance: float
            if not isinstance(perplexity_tolerance, float):
                raise ValueError("perplexity_tolerance must be of float type")
            elif perplexity_tolerance in invalid_numbers:
                raise ValueError("perplexity_tolerance must be finite and not NaN")
            elif perplexity_tolerance < 0:
                raise ValueError("perplexity_tolerance must be a positive number or 0")
        
        # Metric
        if metric is not None: # metric: str
            if not isinstance(metric, str):
                raise ValueError("metric must be of str type")
            elif metric not in accepted_metrics:
                    raise ValueError("Only metrics accepted are 'euclidean' and 'precomputed'")
        
        # Init
        if init is not None:
            if isinstance(init, str):
                if init.lower() not in accepted_inits:
                    raise ValueError("Only init_method values accepted are 'random', 'precomputed' and 'pca'")
                elif metric is not None:
                    if init.lower()=="pca" and metric.lower()=="precomputed":
                        raise ValueError("With init cannot be 'pca' when metric is 'precomputed'")
            elif _is_array_like(init):
                if isinstance(init, Sequence):
                    evaluation = np.array(init)
                else:
                    evaluation = init
                if not isinstance(evaluation.dtype, np.number):
                    raise ValueError("Data type of the initial embedding must be a number")
                elif np.inf in init or np.nan in init:
                    raise ValueError("The initial embedding must not contain NaN or an infinite number")
                elif n_components is not None and evaluation.shape[1]!=n_components:
                    raise ValueError("The initial embedding must have the number of dimensions provided")

            else:
                raise ValueError("init must be a str or ArrayLike")
        
        # Early exaggeration
        if early_exaggeration is not None: # early_exaggeration: float
            if not isinstance(early_exaggeration, (int, float)):
                raise ValueError("early_exaggeration must be a number")
            elif early_exaggeration in invalid_numbers:
                raise ValueError("early_exaggeration must be finite and not NaN")
            elif early_exaggeration <=0.:
                raise ValueError("early_exaggeration must be positive")
        
        # Learning rate
        if learning_rate is not None: # learning_rate: float
            if isinstance(learning_rate, str):
                if learning_rate != "auto":
                    raise ValueError("The only str value acceptable for learning_rate is 'auto'")
            elif isinstance(learning_rate, (int, float)):
                if learning_rate in invalid_numbers:
                    raise ValueError("learning_rate must be finite and not NaN")
                elif learning_rate <=0.:
                    raise ValueError("learning_rate must be positive")
            else:
                raise ValueError("learning_rate must a number or 'auto'")
        
        # Max iter
        if n_iter is not None: # n_iter: int
            if not isinstance(n_iter, int):
                raise ValueError("n_iter must be an integer")
            elif n_iter in invalid_numbers:
                raise ValueError("n_iter must be finite and not NaN")
            elif n_iter <1:
                raise ValueError("n_iter must be a positive number")
        
        # Starting Momentum
        if starting_momentum is not None: # starting_momentum: float
            if not isinstance(starting_momentum, float):
                raise ValueError("starting_momentum must be of float type")
            elif starting_momentum in invalid_numbers:
                raise ValueError("starting_momentum cannot be NaN or inf")
            elif starting_momentum<=0 or starting_momentum>=1:
                raise ValueError("starting_momentum must be strictly in the range (0, 1)")
        
        # Ending Momentum
        if ending_momentum is not None: # ending_momentum: float
            if not isinstance(ending_momentum, float):
                raise ValueError("ending_momentum must be of float type")
            elif ending_momentum in invalid_numbers:
                raise ValueError("ending_momentum cannot be NaN or inf")
            elif ending_momentum<=0 or starting_momentum>=1:
                raise ValueError("ending_momentum must be strictly in the range (0, 1)")
        
        # Momentum threshold
        if momentum_threshold is not None: # momentum_threshold: int
            if not isinstance(momentum_threshold, int):
                raise ValueError("momentum_threshold must be an integer")
            elif momentum_threshold not in range(n_iter):
                raise ValueError("momentum_threshold must be in range(0, n_iter)")
        
        # Seed
        if seed is not None: # seed: int
            if not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            elif seed<0:
                raise ValueError("seed must be positive")
        
        # Verbose
        if verbose is not None: # verbose: int
            if not isinstance(verbose, int):
                raise ValueError("verbose must be an integer")
            elif verbose<0:
                raise ValueError("verbose must be positive")
        
        # Iters check
        if iters_check is not None: #iters_check: int
            if not isinstance(iters_check, int):
                raise ValueError("iters_check must be an integer")
            elif iters_check<1:
                raise ValueError("iters_check must be at least 1")
            elif iters_check>n_iter:
                raise ValueError("iters_check cannot be greater than n_iter")
    def __input_validation(self, input, labels=None):
        if _is_array_like(input):
            result = np.array(input)
        else:
            raise ValueError("The given input is not array-like")

        if self.metric=="precomputed":
            if result.ndim!=2:
                raise ValueError("When metric is 'precomputed', input data must be a square distance matrix")
            elif result.shape[0]!=input.shape[1]:
                raise ValueError("When metric is 'precomputed', input data must be a square distance matrix")
        
        minimum_samples = 2*self.n_neighbors
        if len(result)<minimum_samples:
            raise ValueError("Not enough samples. The given perplexity requires at least {} samples".format(minimum_samples))
        if self.init is not None:
            if _is_array_like(self.init):
                if len(input) != self.init.shape[0]:
                    raise ValueError("The input data must have the same number of samples as the given embedding")
        if labels is not None:
            if not _is_array_like(input):
                raise ValueError("The given input is not array-like")
            else:
                marker_paths = []
                for s in self.plotting_marker_options:
                    m = MarkerStyle(s)
                    marker_paths.append(m.get_path().transformed(m.get_transform()))

                plotting_labels = np.array(labels, dtype=np.str_)
                if plotting_labels.ndim!=1:
                    raise ValueError("labels must be a 1D array")
                else:
                    if np.all(np.char.isnumeric(plotting_labels)): #strings de numeros
                        # np.min_scalar_type
                        aux = np.array(labels, dtype=int)
                        self.plotting_labels = aux
                        self.plotting_colors = aux
                        # self.plotting_markers = np.empty_like(plotting_labels)
                        self.plotting_marker_result = np.empty_like(plotting_labels)
                        self.plotting_markers = np.empty_like(plotting_labels, dtype=Path)
                        unicos = np.unique(aux)
                        if unicos.shape[0]>5:
                            for i in range(unicos.shape[0]):
                                indices = np.argwhere(aux==unicos[i])
                                ind_mark = int(i%len(marker_paths))
                                self.plotting_markers[indices] = marker_paths[ind_mark]
                                self.plotting_marker_result[indices] = self.plotting_marker_options[ind_mark]
                        else:
                            aux2 = MarkerStyle('o')
                            self.plotting_markers.fill(aux2.get_path().transformed(aux2.get_transform()))
                            self.plotting_marker_result.fill('o')
                        
                    else: #strings genericos
                        self.plotting_labels = plotting_labels
                        # self.plotting_markers = np.empty_like(labels, dtype=MarkerStyle)
                        self.plotting_marker_result = np.empty_like(plotting_labels)
                        self.plotting_markers = np.empty_like(labels, dtype=Path)
                        self.plotting_colors = np.empty_like(labels, dtype=int)
                        
                        
                        elementos = np.unique(labels)
                        n_unique = elementos.shape[0]
                        
                        if n_unique<=5:
                            for i in range(n_unique):
                                indices = np.argwhere(plotting_labels==elementos[i])
                                self.plotting_colors[indices] = i
                            aux = MarkerStyle('o')
                            self.plotting_markers.fill(aux.get_path().transformed(aux.get_transform()))
                            self.plotting_marker_result.fill('o')
                        else:
                            n = max(np.floor(np.sqrt(n_unique)), 5)
                            colores = np.array(range(n))
                            for i in range(n_unique):
                                indices = np.argwhere(plotting_labels==elementos[i])
                                self.plotting_colors[indices] = colores[np.divmod(i, n)[0]]
                                self.plotting_markers[indices] = marker_paths[int(i%len(marker_paths))]
                                self.plotting_marker_result[indices] = self.plotting_marker_options[int(i%len(marker_paths))]
        else:
            self.plotting_marker_result = np.full(shape=len(result), fill_value='o')
            self.plotting_colors = np.full(shape=len(result), fill_value=1)
        if result.ndim>2:
            return result.reshape((len(result), np.prod(result.shape[1:])))
        return result
    def __rand_embed(self, input, n_dimensions) -> np.ndarray:
        assert n_dimensions is not None
        n_samples = len(input)
        if self.init is None:
            return self.rng.standard_normal(size=(n_samples, n_dimensions))
        elif isinstance(self.init, str):
            if self.init.lower()=="pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_dimensions, svd_solver="randomized", random_state=self.seed)
                pca.set_output(transform="default")
                data_embedded = pca.fit_transform(input).astype(np.float32, copy=False)
                return data_embedded / np.std(data_embedded[:, 0]) * 1e-4
            else: #init = "random"
                return self.rng.standard_normal(size=(n_samples, n_dimensions))
        else:
            return self.init.copy()
    
    def fit(self, input, labels=None, return_last=False) -> np.ndarray:
        """Fit the given data and perform the embedding

        Parameters
        ----------
        X: array of shape (n_samples, n_features).
            The data to fit.
        classes: 1-D array of size (n_samples). Optional.
            Array with the class that each element in X belongs to.
        """

        #====Tiempo de inicio para verbosidad====================================================================================================================
        t0 = time.time_ns()

        #====Input con dimensiones correctas=====================================================================================================================
        X = self.__input_validation(input, labels)


        self.init_embed = self.__rand_embed(X, self.n_components)
        
        #====Ajuste del learning rate============================================================================================================================
        if self.learning_rate == "auto":
            self.lr = len(X) / self.early_exaggeration
            self.lr = np.maximum(self.lr, 50)
        else:
            self.lr = self.learning_rate

        #====Obtener P===========================================================================================================================================
        if self.metric=="precomputed":
            p = similarities.joint_probabilities_gaussian(X, self.perplexity, self.perplexity_tolerance)
        else:
            dist_original = similarities.pairwise_euclidean_distance(X) #solo es necesario para calcular P
            p = similarities.joint_probabilities_gaussian(dist_original, self.perplexity, self.perplexity_tolerance)
            del dist_original #dist_original ya no hace falta
        
        #====Descenso de gradiente===============================================================================================================================
        
        # inicializar self.update, self.current_embed
        self.update = np.zeros_like(self.init_embed)
        self.best_embed = self.init_embed.copy()
        self.current_embed = self.init_embed.copy()

        # if self.n_components==2:
        #     line = self.plotting_ax.scatter(self.init_embed.T[0], self.init_embed.T[1], marker=MarkerStyle(self.plotting_markers), c=self.plotting_colors, s=self.plotting_size)
        # else:
        #     line = self.plotting_ax.scatter(self.init_embed.T[0], self.init_embed.T[1], self.init_embed.T[2], marker=MarkerStyle(self.plotting_markers), c=self.plotting_colors, s=self.plotting_size)
        

        
        if self.n_components==2:
            line = self.plotting_ax.scatter(self.init_embed.T[0], self.init_embed.T[1], label=self.plotting_labels, c=self.plotting_colors, s=self.plotting_size)
        else:
            line = self.plotting_ax.scatter(self.init_embed.T[0], self.init_embed.T[1], self.init_embed.T[2], label=self.plotting_labels, c=self.plotting_colors, s=self.plotting_size)
        if self.plotting_labels is not None:
            leg = self.plotting_ax.legend(*line.legend_elements(), loc="lower right")
            self.plotting_ax.add_artist(leg)
        ani = animation.FuncAnimation(self.plotting_fig, self.__update_anim, self.n_iter, fargs=[p, self.plotting_ax], interval=100, repeat=False)
        plt.show()
        # result = self.__gradient_descent(self.n_iter, p*self.early_exaggeration, return_last)
        
        
        #====Salida por consola de verbosidad====================================================================================================================
        if self.verbose>0:
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
        
        self.embedding_finished = True
        return self.current_embed
    
    def __gradient_descent(self, t, p, return_last=False):
        #==== Parametros extra ===================================================================================================================================
        iter_threshold = self.momentum_threshold
        momentum = self.momentum_start
        
        
        embed = self.init_embed.copy()
        # previous_embed = np.zeros(self.init_embed.shape, dtype=embed.dtype) # y(t-2)
        update = np.zeros_like(embed)
        best_embed_ = np.zeros_like(embed).flatten()
        verbos = self.verbose
        check_i = self.iters_check
        for i in range(0, t+1):
            if verbos>1 and i%check_i==0:
                print("---------------------------------")
                print("Comenzando iteracion {}/{}".format(i,t))
            

            embed_dist = similarities.pairwise_euclidean_distance(embed)
            q = similarities.joint_probabilities_student(embed_dist)
            
            
            #==== cost ==============================================================================================================================================
            if i%check_i==0 or i==t:
                cost = kl_divergence(p, q)
                if verbos>1:
                    print("KL(t={}) = {:.6f}".format(i, cost))
                if self.embed_cost is None or cost<self.embed_cost:
                    self.embed_i = i
                    self.embed_cost = cost
                    best_embed_ = embed.flatten()

            #==== momentum change ===================================================================================================================================
            if i==iter_threshold:
                p /= self.early_exaggeration
                momentum = self.momentum_end
            elif i==t:
                break
            #==== grad ==============================================================================================================================================
            grad = gradient(p, q, embed, embed_dist)

            #==== embed update ======================================================================================================================================
            # update = momentum*(embed-previous_embed) - grad*self.lr
            update = momentum*update - grad*self.lr
            # previous_embed = embed.copy()
            embed += update
            # del grad, update
            del grad
            # gc.collect() #liberar memoria despues de cada iteracion

        if return_last:
            self.embed_i = t-1
            self.embed_cost = cost
            return embed
        else:
            return best_embed_.reshape(self.init_embed.shape, order='C')

    def __update_embed(self, i, affinities):
        if self.verbose>1 and i%self.iters_check==0:
            print("---------------------------------")
            print("Comenzando iteracion {}/{}".format(i,self.n_iter))

        embed_dist = similarities.pairwise_euclidean_distance(self.current_embed)
        q = similarities.joint_probabilities_student(embed_dist)


        # Cambio de momento
        if i<self.momentum_threshold:
            p = self.early_exaggeration*affinities
            momentum = self.momentum_start
        else:
            p = affinities
            momentum = self.momentum_end
        
        # Coste
        if i%self.iters_check==0:
            cost = kl_divergence(p, q)
            if self.verbose>1:
                    print("KL(t={}) = {:.6f}".format(i, cost))
            if self.embed_cost is None or cost<self.embed_cost:
                self.embed_i = i
                self.embed_cost = cost
                self.best_embed = self.current_embed.flatten()

        # Calculo de nuevo embed
        grad = gradient(p, q, self.current_embed, embed_dist)
        self.update = momentum*self.update - grad*self.lr
        self.current_embed += self.update


    def __update_anim(self, i, affinities, ax:Axes):
        self.__update_embed(i, affinities)
        
        #===Plotting===============================================================================
        ax.clear()
        
        x = self.current_embed.T[0]
        y = self.current_embed.T[1]
        if self.n_components==3:
            z = self.current_embed.T[2]
        
        leg_aux1 = []
        leg_aux2 = []

        for m in np.unique(self.plotting_marker_result):
            cond = self.plotting_marker_result==m
            l = None if self.plotting_labels is None else self.plotting_labels[cond]
            if self.n_components==2:
                line = ax.scatter(x[cond], y[cond], marker=m, c=self.plotting_colors[cond], label=l, s=self.plotting_size)
            else:
                line = ax.scatter(x[cond], y[cond], z[cond], marker=m, c=self.plotting_colors[cond], label=l, s=self.plotting_size)
            
            # PARA LA LEYENDA
            aux1, aux2 = line.legend_elements()
            leg_aux1.extend(aux1)
            leg_aux2.extend(aux2)

        # Labels
        if self.plotting_labels is not None:
            leg = ax.legend(leg_aux1, leg_aux2, loc="lower right")
            ax.add_artist(leg)

        # Title
        plt.title("Current Iteration: {}/{} \n Best cost: i={}, cost={:.3f}".format(i+1, self.n_iter, self.embed_i, self.embed_cost))
        if i>10:
            print("todo bien")
            exit(0)

    def get_embedding_cost_info(self):
        assert self.embedding_finished
        return self.embed_i, self.embed_cost

    

    