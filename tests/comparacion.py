import numpy as np
import mytsnelib.utils as ut


#======================================================#
#---Parametros para entrenar el modelo-----------------#
n_dimensions = 2
perplexity = 30.
perplexity_tolerance = 1e-10
metric = "euclidean"
init_method = "random"
init_embed = None
early_exaggeration = 12.
lr = 200.
max_iter = 10
momentum_params = [250., 0.5, 0.8]
seed = 4
iters_check = 50
#---Cosas que mostrar por consola----------------------#
nivel_verbose=0
#======================================================#

#===Mio======================================================================#
def probar_mio(data, labels, *, display=False, title=None, calcular_coste=False, display_best_cost=False, medir_rendimiento=False, print_cost_history=False):
    from mytsnelib import functions
    model = functions.TSne(n_dimensions=n_dimensions,
                        perplexity=perplexity,
                        perplexity_tolerance=perplexity_tolerance,
                        early_exaggeration=early_exaggeration,
                        learning_rate=lr,
                        max_iter=max_iter,
                        momentum_params=momentum_params,
                        seed=seed,
                        verbose=nivel_verbose,
                        iters_check=iters_check)
    data_embedded = model.fit(data,classes=labels, compute_cost=calcular_coste, measure_efficiency=medir_rendimiento)
    
    if display:
        if calcular_coste and display_best_cost:
            indice = np.argmin(model.cost_history)
            embed = model.cost_history_embed[indice]
            if indice<len(model.cost_history)-1:
                title = "Iteración con mejor coste: {}/{}".format(max(0, model.iters_check*(indice-1)), max_iter)
            else:
                title = "Iteración con mejor coste: {}".format(max_iter)
        else:
            title = "Mostrando embedding final"
            embed = np.copy(data_embedded)
        ut.display_embed(embed, labels, title=title)
    
    if print_cost_history:
        historia_coste = np.array(model.cost_history)
        if len(historia_coste)>0:
            print("Cost history:")
            print(historia_coste.__str__())
            for i in range(1, len(historia_coste)):
                if historia_coste[i]>historia_coste[i-1]:
                    print("--------------------------------------------------------------")
                    print("El coste en el indice {} es mayor que el previo, indice {}".format(i, i-1))
                    print("Coste con indice {}: {}".format(i-1, historia_coste[i-1]))
                    print("Coste con indice {}: {}".format(i, historia_coste[i]))
            print("--------------------------------------------------------------")
            print("Tamaño del historial de costes: {}".format(len(historia_coste)))
            print("Coste minimo, con indice {}: {}".format(np.argmin(historia_coste), np.min(historia_coste)))

#===Scikit-learn=============================================================#
def probar_sklearn(data, labels, *, display=False, title=None):
    import sklearn.manifold as mnf
    model = mnf.TSNE(n_components=n_dimensions,
                     learning_rate='auto',
                     init='random',
                     perplexity=perplexity,
                     verbose=nivel_verbose)
    data_embedded = model.fit_transform(data)

    if display:
        ut.display_embed(data_embedded, labels, title=title)

#===PCA======================================================================#
def probar_pca(data, labels, *, display=False, title=None):
    from sklearn.decomposition import PCA
    random_state = np.random.RandomState(seed)
    pca = PCA(n_components=n_dimensions, svd_solver="randomized", random_state=random_state)
    # Always output a numpy array, no matter what is configured globally
    pca.set_output(transform="default")
    data_embedded = pca.fit_transform(data).astype(np.float32, copy=False)
    # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
    # the default value for random initialization. See issue #18018.
    data_embedded = data_embedded / np.std(data_embedded[:, 0]) * 1e-4

    if display:
        ut.display_embed(data_embedded, labels, title=title)

#===Autoencoder==============================================================#
def probar_autoencoder(train_data, test_data, test_labels, *, display=False, title=None, display_amount=-1):
    import tensorflow as tf
    from keras.api import Sequential
    from keras.api.losses import MeanSquaredError
    from keras.api.layers import Dense, Flatten, Reshape
    from keras.api.models import Model

    class Autoencoder(Model):
        def __init__(self, latent_dim, shape):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = Sequential([Flatten(), Dense(latent_dim, activation='relu'),])
            self.decoder = Sequential([Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'), Reshape(shape)])
        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    shape = test_data.shape[1:]
    autoencoder = Autoencoder(n_dimensions, shape)

    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    autoencoder.fit(train_data, train_data, epochs=max_iter, shuffle=True, validation_data=(test_data, test_data), verbose=nivel_verbose)

    data_embedded = autoencoder.encoder(test_data).numpy()

    if display:
        if display_amount>0:
            rand_indices = np.random.randint(low=0, high=len(data_embedded), size=display_amount)
            display_data = data_embedded[rand_indices]
            display_labels = test_labels[rand_indices]
            ut.display_embed(display_data, display_labels, title=title)
        else:
            ut.display_embed(data_embedded, test_labels, title=title)


#=======================================================================================================#
#===========================TODO: terminar de implementar el metodo de prueba===========================#
#=======================================================================================================#
def probar_bht(data, labels, *, verbose=1, display=False, title=None):
    import bhtsne
    embedding_bht = bhtsne.tsne(data, initial_dims=data.shape[1])
    

def probar_open(data, labels, *, verbose=1, display=False, title=None):
    import openTSNE
    embedding_open = openTSNE.TSNE(n_iter=max_iter, n_components=n_dimensions, perplexity=perplexity).fit(data)
#=======================================================================================================#
#=======================================================================================================#
#=======================================================================================================#
