import pickle
import tqdm
from absl import flags
import numpy as np
import tensorflow as tf
import embed
from sklearn.neighbors import KNeighborsRegressor

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', "AntMorphology",
                    'which task to evaluate')
flags.DEFINE_integer('k_nn', 10,
                    'which task to evaluate')
flags.DEFINE_integer('top_p', 20,
                    'top data percentile')
def normalize(x):
    #normalize data
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    return (x - x_mean) / x_std

def get_embed_model_vpn(state_dim, action_dim, latent_dim):
    embed_model = embed.VpnLearner(
        state_dim,
        action_dim,
        embedding_dim=latent_dim,
        sequence_length=8,
        learning_rate=None)
    return embed_model 


def load_embedded_data(dataset_name):
    with open("embedding_datasets/"+dataset_name +'_embedding.pickle', 'rb') as f:
            dataset = pickle.load(f)    

    return dataset["x"], dataset["y"]


def embed_designs(x, embed_model):
    x = tf.convert_to_tensor(x)
    x_embedding = embed_model(x).numpy()
    
    return x_embedding

def osel_evaluate_designs(designs, neigh):
    preds = neigh.predict(designs)
    max_v = np.max(preds)
    # med_v = np.median(preds)
    return max_v #, med_v


if __name__ == '__main__':
    k_nn = FLAGS.k_nn
    top_p = FLAGS.top_p
    dataset_name = FLAGS.dataset_name #["AntMorphology", "DKittyMorphology", "Superconductor", "TFBind8", "UTR"]
    task_to_dim = {"AntMorphology" : 60, 
                   "DKittyMorphology" : 56, 
                   "Superconductor" : 86,
                   "TFBind8" : 32, 
                   "UTR": 32}
    if dataset_name in ["TFBind8", "UTR"]:
        latent_dim = 8
    else:
        latent_dim = 32

    x, y = load_embedded_data(dataset_name)
    neigh = KNeighborsRegressor(n_neighbors=k_nn)
    x = np.nan_to_num(x)
    neigh.fit(x, y)  

    embed_model = get_embed_model_vpn(task_to_dim[dataset_name], task_to_dim[dataset_name], latent_dim)
    embed_model.load_weights("embedding_models/"+dataset_name+'_embedder/')
    all_designs = "designs/"+dataset_name+"_designs"
    with open(dataset_name+'_designs_top_'+str(100 - top_p)+'.pickle', 'rb') as f:
            all_designs = pickle.load(f) 
            
    epochs_ = [50 * (i+1) for i in range(8)]
    osel_scores = []
    for epoch in epochs_:
        designs = all_designs[epoch]
        score = osel_evaluate_designs(designs, neigh)
        osel_scores.append(score)

    with open(dataset_name+"_osel_scores_top_"+str(100 - top_p), "wb") as f:
        pickle.dump(osel_scores, f)

    

    