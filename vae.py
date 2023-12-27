from data import build_pipeline
from logger import Logger
from coms_cleaned.trainers import VAETrainer
from coms_cleaned.nets import SequentialVAE
import tensorflow as tf
import numpy as np
import pickle

task_name = ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 
                'UTR-ResNet-v0']

def train_vae(
        task,
        task_name="tfbind8",
        vae_hidden_size=64,
        vae_latent_size=32,
        vae_activation='relu',
        vae_kernel_size=3,
        vae_num_blocks=4,
        vae_lr=0.0003,
        vae_beta=1.0,
        vae_batch_size=32,
        vae_val_size=200,
        vae_epochs=50,
        ):
    
    logging_dir = task_name + "_" + str(vae_latent_size)
    logger = Logger(logging_dir)

    x = task.x
    y = task.y

    vae_model = SequentialVAE(
        task, hidden_size=vae_hidden_size,
        latent_size=vae_latent_size, activation=vae_activation,
        kernel_size=vae_kernel_size, num_blocks=vae_num_blocks)

    vae_trainer = VAETrainer(
        vae_model, optim=tf.keras.optimizers.Adam,
        lr=vae_lr, beta=vae_beta)

    train_data, val_data = build_pipeline(
        x=x, y=y, batch_size=vae_batch_size,
        val_size=vae_val_size)

    # estimate the number of training steps per epoch
    vae_trainer.launch(train_data, val_data,
                        logger, vae_epochs)

    # map the x values to latent space
    x = vae_model.encoder_cnn.predict(x)[0]
    vae_model.decoder_cnn.predict(x)[0]

    vae_model.decoder_cnn.save("decoder/" + task_name + "_" + str(vae_latent_size)+"/")
    vae_model.encoder_cnn.save("encoder/" + task_name + "_" + str(vae_latent_size)+"/")

    mean = np.mean(x, axis=0, keepdims=True)
    standard_dev = np.std(x - mean, axis=0, keepdims=True)
    x = (x - mean) / standard_dev
    
    train_data = dict()
    train_data["x"] = x
    train_data["y"] = y
    train_data["mean"] = mean
    train_data["std_dev"] = standard_dev 
    with open(task_name.split("-")[0]+'_'+str(vae_latent_size)+'.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return vae_model


