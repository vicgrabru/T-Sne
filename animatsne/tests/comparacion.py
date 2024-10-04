import numpy as np
from src.animatsne import utils as ut
import time


#======================================================#
#---Parametros para entrenar el modelo-----------------#
n_dimensions = 2
perplexity = 50
lr = "auto"
# lr = 500
early_exaggeration = 12
max_iter = 5000
seed = 4
# seed = int(time.time())
iters_check = 50
init = "random"
fraccionDatosVecinos = 3
start_momentum = 0.5
end_momentum = 0.8
momentum_threshold = 250


argumentos_globales_modelos = {
    "perplexity": 50
}


argumentos_modelo_mio = {
    "n_dimensions": 2,
    "perplexity": 50,
    "perplexity_tolerance": 1e-10,
    "early_exaggeration": 12,
    "learning_rate": "auto",
    "init": "random",
    "max_iter": 5000,
    "starting_momentum":  0.5,
    "ending_momentum":  0.8,
    "momentum_threshold": 250,
    "seed": 4,
    "verbose": 1,
    "iters_check": 50,
}

argumentos_modelo_skl = {
    "n_components": 2,
    "learning_rate": "auto",
    "init": "random",
    "perplexity": 50,
    "early_exaggeration": 12,
    "verbose": 1,
}

argumentos_modelo_open = {
    "n_iter": 5000,
    "n_components": 2,
    "perplexity": 50,
    "early_exaggeration": 12,
    "early_exaggeration_iter": 250,
    "initial_momentum": 0.5,
    "final_momentum": 0.8,
    "learning_rate": "auto",
    "verbose": 1,

}


#---Cosas que mostrar por consola----------------------#
nivel_verbose=0
#======================================================#

#===Mio======================================================================#
def probar_mio(data, labels, *, display=False, title=None, print_tiempo=False, trust=False):
    if print_tiempo:
        argumentos_modelo_mio["verbose"] = 0
    verbosidad = 0 if print_tiempo else nivel_verbose
    from src.animatsne import animatsne
    perplexity_tolerance = 1e-10
    
    t0 = time.time_ns()
    model = animatsne.TSne(**argumentos_modelo_mio, **argumentos_globales_modelos)
    data_embedded = model.fit(data, labels)
    t_diff = (time.time_ns()-t0)*1e-9
    
    if print_tiempo:
        ut.print_tiempo(t_diff, metodo="Mio")
    if trust:
        ut.print_trust(data, data_embedded, "mio")
    if display:
        best_iter, best_cost = model.get_embedding_cost_info()
        title = "Mostrando embedding con mejor coste, en la iteracion {}/{}".format(best_iter, max_iter)
        embed = np.copy(data_embedded)
        ut.display_embed(embed, labels, title=title)
        del embed,title

    del t_diff,t0
    del data_embedded,model,perplexity_tolerance,verbosidad

#===Scikit-learn=============================================================#
def probar_sklearn(data, labels, *, display=False, title=None, print_tiempo=False, trust=False):
    import sklearn.manifold as mnf
    verbosidad = 0 if print_tiempo else nivel_verbose
    t0 = time.time_ns()
    model = mnf.TSNE(verbose=verbosidad, **argumentos_modelo_skl)

    data_embedded = model.fit_transform(data)
    t_diff = (time.time_ns()-t0)*1e-9
    if print_tiempo:
        ut.print_tiempo(t_diff, metodo="Scikit-learn")
    if trust:
        ut.print_trust(data, data_embedded, "Scikit-Learn")
    if display:
        ut.display_embed(data_embedded, labels, title=title)

#===openTSNE=================================================================#
def probar_open(data, labels, *, verbose=1, display=False, title=None, print_tiempo=False, trust=False):
    import openTSNE
    verbosidad = 0 if print_tiempo else nivel_verbose
    t0 = time.time_ns()

    model = openTSNE.TSNE(n_iter=max_iter,
                          n_components=n_dimensions,
                          perplexity=perplexity,
                          early_exaggeration=early_exaggeration,
                          early_exaggeration_iter=momentum_threshold,
                          initial_momentum=start_momentum,
                          final_momentum=end_momentum,
                          learning_rate=lr,
                          verbose=verbosidad)
    data_embedded = model.fit(data)
    t_diff = (time.time_ns()-t0)*1e-9
    if print_tiempo:
        ut.print_tiempo(t_diff, metodo="openTSNE")
    if trust:
        ut.print_trust(data, data_embedded, metodo="openTSNE")
    if display:
        ut.display_embed(data_embedded, labels, title=title)
    


#===PCA======================================================================#
def probar_pca(data, labels, *, display=False, title=None, print_tiempo=False, trust=False):
    from sklearn.decomposition import PCA
    t0 = time.time_ns()
    pca = PCA(n_components=n_dimensions, svd_solver="randomized", random_state=seed)
    # Always output a numpy array, no matter what is configured globally
    pca.set_output(transform="default")
    data_embedded = pca.fit_transform(data).astype(np.float32, copy=False)
    # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
    # the default value for random initialization. See issue #18018.
    data_embedded = data_embedded / np.std(data_embedded[:, 0]) * 1e-4
    t_diff = (time.time_ns()-t0)*1e-9
    if print_tiempo:
        ut.print_tiempo(t_diff, metodo="PCA")
    if trust:
        ut.print_trust(data, data_embedded, metodo="PCA")
    if display:
        ut.display_embed(data_embedded, labels, title=title)
    del t_diff,t0,data_embedded,pca

#===Autoencoder==============================================================#
def probar_autoencoder(train_data, test_data, test_labels, *, display=False, title=None, display_amount=-1, print_tiempo=False, trust=False):
    from keras.api import Sequential
    from keras.api.layers import Dense
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
    sgd = SGD(learning_rate=0.02)
    t0 = time.time_ns()
    autoencoder.compile(optimizer='sgd', loss='mse')
    autoencoder.fit(train_data, train_data, epochs=10, shuffle=True, validation_data=(test_data, test_data), verbose=nivel_verbose)
    t_diff = (time.time_ns()-t0)*1e-9
    
    
    rand_indices = np.random.randint(low=0, high=len(test_data), size=display_amount)
    if display_amount>0:
        display_data = test_data[rand_indices]
        display_labels = test_labels[rand_indices]
    else:
        display_data = test_data
        display_labels = test_labels
    
    data_embedded = autoencoder.encoder(display_data).numpy()

    if print_tiempo:
        ut.print_tiempo(t_diff, metodo="Autoencoders")
    
    if trust:
        ut.print_trust(display_data, data_embedded, metodo="Autoencoder")

    if display:
        ut.display_embed(data_embedded, display_labels, title=title)
    
    del data_embedded,display_data,display_labels,rand_indices,t_diff,autoencoder,t0,sgd,shape


#=======================================================================================================#
#===========================TODO: terminar de implementar el metodo de prueba===========================#
#=======================================================================================================#
def probar_bht(data, labels, *, verbose=1, display=False, title=None):
    import bhtsne
    embedding_bht = bhtsne.tsne(data, initial_dims=data.shape[1])
#=======================================================================================================#
#=======================================================================================================#
#=======================================================================================================#

