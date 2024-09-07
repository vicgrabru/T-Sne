import numpy as np
import mytsnelib.utils as ut


#======================================================#
#---Parametros para entrenar el modelo-----------------#
n_dimensions = 2
# perplexity = 50
lr = "auto"
early_exaggeration = 12.
max_iter = 10000
seed = 4
iters_check = 50
init_method = "random"
#---Cosas que mostrar por consola----------------------#
nivel_verbose=1
#======================================================#

#===Mio======================================================================#
def probar_mio(data, labels, *, display=False, title=None, calcular_coste=False, display_best_cost=False, print_cost_history=False):
    from mytsnelib import functions
    #perp = n_vecinos*3
    perp = np.floor(len(data)/3)
    perplexity_tolerance = 1e-10
    momentum_params = [250, 0.5, 0.8]
    model = functions.TSne(n_dimensions=n_dimensions,
                        perplexity=perp,
                        perplexity_tolerance=perplexity_tolerance,
                        early_exaggeration=early_exaggeration,
                        learning_rate=lr,
                        max_iter=max_iter,
                        momentum_params=momentum_params,
                        seed=seed,
                        verbose=nivel_verbose,
                        iters_check=iters_check)
    data_embedded = model.fit(data, compute_cost=calcular_coste)
    
    if display:
        if calcular_coste and display_best_cost:
            title = "Iteración con mejor coste: {}/{}".format(model.best_cost_iter, max_iter)
            embed = np.copy(model.best_cost_embed)
        else:
            title = "Mostrando embedding final"
            embed = np.copy(data_embedded)
        ut.display_embed(embed, labels, title=title)
    
    if print_cost_history and calcular_coste:
        historia_coste = np.array(model.cost_history)
        # print("Cost history:")
        # print("{}".format(historia_coste))
        # for i in range(1, len(historia_coste)):
        #     if historia_coste[i]>historia_coste[i-1]:
        #         print("-------------------------------------------------------------------")
        #         print("Coste {} > Coste {}".format(i+1, i))
        #         print("Coste {}: {}".format(i+1, historia_coste[i]))
        #         print("Coste {}: {}".format(i, historia_coste[i-1]))
        # print("-------------------------------------------------------------------")
        print("Coste minimo, {}/{}: {}".format(np.argmin(historia_coste)+1, len(historia_coste), np.min(historia_coste)))
        print("Ultimo coste: {}".format(historia_coste[-1]))
        print("===================================================================")

#===Scikit-learn=============================================================#
def probar_sklearn(data, labels, *, display=False, title=None):
    import sklearn.manifold as mnf
    perp = np.floor(len(data)/3)
    model = mnf.TSNE(n_components=n_dimensions,
                     learning_rate=lr,
                     init=init_method,
                     perplexity=perp,
                     early_exaggeration=early_exaggeration,
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
            self.encoder = Sequential([
                Dense(100*latent_dim, activation='relu'),
                Dense(50*latent_dim, activation='relu'),
                Dense(latent_dim, activation='relu')
            ])
            self.decoder = Sequential([
                Dense(50*latent_dim, activation='relu'),
                Dense(100*latent_dim, activation='relu'),
                Dense(shape[0], activation='sigmoid')
            ])
        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    shape = test_data.shape[1:]
    autoencoder = Autoencoder(n_dimensions, shape)

    # autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    
    from keras.api.optimizers import SGD
    sgd = SGD(learning_rate=lr/10000)
    autoencoder.compile(optimizer='sgd', loss='mse')
    autoencoder.fit(train_data, train_data, epochs=10, shuffle=True, validation_data=(test_data, test_data), verbose=nivel_verbose)
    
    if display_amount>0:
        rand_indices = np.random.randint(low=0, high=len(test_data), size=display_amount)
        display_data = test_data[rand_indices]
        display_labels = test_labels[rand_indices]
    else:
        display_data = test_data
        display_labels = test_labels
    
    data_embedded = autoencoder.encoder(display_data).numpy()

    if display:
        ut.display_embed(data_embedded, display_labels, title=title)


#=======================================================================================================#
#===========================TODO: terminar de implementar el metodo de prueba===========================#
#=======================================================================================================#
def probar_bht(data, labels, *, verbose=1, display=False, title=None):
    import bhtsne
    embedding_bht = bhtsne.tsne(data, initial_dims=data.shape[1])
    

def probar_open(data, labels, *, verbose=1, display=False, title=None):
    import openTSNE
    perp = np.floor(len(data)/9)
    embedding_open = openTSNE.TSNE(n_iter=max_iter, n_components=n_dimensions, perplexity=perp).fit(data)
#=======================================================================================================#
#=======================================================================================================#
#=======================================================================================================#
