import numpy as np


from mytsnelib import functions
import sklearn.manifold as mnf
import mytsnelib.utils as ut
import time



def test_haversine():
    assert True


#=================================================================#
include_n_samples = 300
index_start = 0

read_csv = ut.read_csv("data/digits.csv", has_labels=True)
data_full = read_csv[0].astype(np.int32)
labels_full = read_csv[1]
data = data_full[index_start:index_start+include_n_samples,:]
labels = labels_full[index_start:index_start+include_n_samples]
#=================================================================#


#======================================================#
#---Parametros para entrenar el modelo-----------------#
n_dimensions = 2
perplexity = 50.
perplexity_tolerance = 1e-10
metric = "euclidean"
init_method = "random"
init_embed = None
early_exaggeration = 12.
learning_rate = 200.
max_iter = 1000
momentum_params = [250., 0.5, 0.8]
seed = 4
iters_check = 50
#---Parametros para calculos extra---------------------#
calcular_coste=False
#---Cosas que mostrar por consola----------------------# 
medir_rendimiento=False
print_cost_history=False
nivel_verbose=0
#---Mostrar el embedding-------------------------------#
display_sklearn = True
mostrar_resultado = "last" # None para no mostrar, "last" para mostrar la ultima, "cost" para mostrar la que obtiene mejor coste
#======================================================#

def probar_mio(data, labels, *, title=None):
    model = functions.TSne(n_dimensions=n_dimensions,
                           perplexity=perplexity,
                           perplexity_tolerance=perplexity_tolerance,
                           max_iter=max_iter,
                           verbose=nivel_verbose,
                           seed=seed,
                           iters_check=iters_check)
    data_embedded = model.fit(data,classes=labels, compute_cost=calcular_coste, measure_efficiency=medir_rendimiento)
    
    if mostrar_resultado is not None:
        match mostrar_resultado:
            case "last":
                title = "Mostrando embedding final"
                embed = np.copy(data_embedded)
            case "cost":
                indice = np.argmin(model.cost_history)
                embed = model.cost_history_embed[indice]
                if indice<len(model.cost_history)-1:
                    title = "Iteración con mejor coste: {}/{}".format(max(0, model.iters_check*(indice-1)), max_iter)
                else:
                    title = "Iteración con mejor coste: {}".format(max_iter)
            case _:
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

def probar_sklearn(data, labels, *, display=False, title=None):
    model = mnf.TSNE(n_components=n_dimensions,
                     learning_rate='auto',
                     init='random',
                     perplexity=perplexity,
                     verbose=nivel_verbose)
    data_embedded = model.fit_transform(data)

    if display:
        ut.display_embed(data_embedded, labels, title=title)
        



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



def probar_otra_cosa():
    print("Probando otra cosa")
    a = b = 1
    print("a: {}".format(a))
    print("b: {}".format(b))
    b = 2
    print("a: {}".format(a))
    print("b: {}".format(b))







# t0 = time.time_ns()
probar_mio(data, labels)
# t_diff_1 = (time.time_ns()-t0)*1e-9
# print("Tiempo de ejecucion mio (s): {}".format(t_diff_1))



# t2 = time.time_ns()
#probar_sklearn(data, labels, display=display_sklearn)
# t_diff_2 = (time.time_ns()-t2)*1e-9
# print("Tiempo de ejecucion skl (s): {}".format(t_diff_2))



