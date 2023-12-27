# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding for state representation learning."""

import typing

# from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# from rl_repr.batch_rl import keras_utils
# from rl_repr.batch_rl import policies
import keras_utils

def soft_update(net, target_net, tau=0.005):
  for var, target_var in zip(net.variables, target_net.variables):
    new_value = var * tau + target_var * (1 - tau)
    target_var.assign(new_value)

def create_mlp(
    input_dim,
    output_dim,
    hidden_dims = (256, 256)):

  relu_gain = tf.math.sqrt(2.0)
  relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
  near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)

  layers = []
  for hidden_dim in hidden_dims:
    layers.append(
        tf.keras.layers.Dense(
            hidden_dim,
            activation=tf.nn.relu,
            kernel_initializer=relu_orthogonal))

  if isinstance(input_dim, int):
    input_shape = (input_dim,)
  else:
    input_shape = input_dim
  inputs = tf.keras.Input(shape=input_dim)
  outputs = tf.keras.Sequential(
      layers + [tf.keras.layers.Dense(
          output_dim - 1, kernel_initializer=near_zero_orthogonal),
                tf.keras.layers.Lambda(
                    lambda x: tf.concat([x, tf.ones_like(x[Ellipsis, :1])], -1)),
                tf.keras.layers.LayerNormalization(
                    epsilon=0.0, center=False, scale=False)]
      )(inputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


class EmbedNet(tf.keras.Model):
  """An embed network."""

  def __init__(self,
               state_dim,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256)):
    """Creates a neural net.

    Args:
      state_dim: State size.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()

    inputs = tf.keras.Input(shape=(state_dim,))
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0
    self.embedder = keras_utils.create_mlp(
        inputs.shape[-1], self.embedding_dim, hidden_dims=hidden_dims,
        activation=tf.nn.swish,
        near_zero_last_layer=bool(num_distributions))

  @tf.function
  def call(self,
           states,
           stop_gradient = True):
    """Returns embeddings of states.

    Args:
      states: A batch of states.
      stop_gradient: Whether to put a stop_gradient on embedding.

    Returns:
      Embeddings of states.
    """
    if not self.num_distributions:
      out = self.embedder(states)
    else:
      all_logits = self.embedder(states)
      all_logits = tf.split(all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [tfp.distributions.Categorical(logits=logits).sample()
                     for logits in all_logits]
      all_onehot_samples = [tf.one_hot(samples, self.embedding_dim // self.num_distributions)
                            for samples in all_samples]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      out = joined_onehot_samples + joined_probs - tf.stop_gradient(joined_probs)

    if stop_gradient:
      return tf.stop_gradient(out)
    return out


class VpnLearner(tf.keras.Model):
  """A learner for value prediction network."""

  def __init__(self,
               state_dim,
               action_dim,
               embedding_dim = 256,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None,
               discount = 0.95,
               tau = 1.0,
               target_update_period = 1000):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
    """
    super().__init__()
    self.input_dim = state_dim
    self.embedding_dim = embedding_dim
    self.sequence_length = sequence_length
    self.discount = discount
    self.tau = tau
    self.target_update_period = target_update_period

    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        hidden_dims=hidden_dims)
    self.f_value = keras_utils.create_mlp(
        self.embedding_dim, 1, hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_value_target = keras_utils.create_mlp(
        self.embedding_dim, 1, hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_trans = keras_utils.create_mlp(
        self.embedding_dim + action_dim, self.embedding_dim,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_out = keras_utils.create_mlp(
        self.embedding_dim + action_dim, 2,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish)

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.all_variables = self.variables
    soft_update(self.f_value, self.f_value_target, tau=1.0)

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: 2 or 3 dimensional state tensors.
      downstream_input_mode: mode of downstream inputs, e.g., state-ctx.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    assert len(states.shape) == 2
    return self.embedder(states, stop_gradient=stop_gradient)

  def fit(self, states, actions,
          rewards, discounts,
          next_states):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.
      rewards: Batch of sequences of rewards.
      next_states: Batch of sequences of next states.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      embeddings = self.embedder(states[:, 0, :], stop_gradient=False)
      all_pred_values = []
      all_pred_rewards = []
      all_pred_discounts = []
      for idx in range(self.sequence_length):
        pred_value = self.f_value(embeddings)[Ellipsis, 0]
        pred_reward, pred_discount = tf.unstack(
            self.f_out(tf.concat([embeddings, actions[:, idx, :]], -1)),
            axis=-1)
        pred_embeddings = embeddings + self.f_trans(
            tf.concat([embeddings, actions[:, idx, :]], -1))

        all_pred_values.append(pred_value)
        all_pred_rewards.append(pred_reward)
        all_pred_discounts.append(pred_discount)

        embeddings = pred_embeddings

      last_value = tf.stop_gradient(self.f_value_target(embeddings)[Ellipsis, 0]) / (1 - self.discount)
      all_true_values = []
      for idx in range(self.sequence_length - 1, -1, -1):
        value = self.discount * discounts[:, idx] * last_value + rewards[:, idx]
        all_true_values.append(value)
        last_value = value
      all_true_values = all_true_values[::-1]

      reward_error = tf.stack(all_pred_rewards, -1) - rewards[:,:,0]
      value_error = tf.stack(all_pred_values, -1) - (1 - self.discount) * tf.stack(all_true_values, -1)
      reward_loss = tf.reduce_sum(tf.math.square(reward_error), -1)
      value_loss = tf.reduce_sum(tf.math.square(value_error), -1)

      loss = tf.reduce_mean(reward_loss + value_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))
    if self.optimizer.iterations % self.target_update_period == 0:
      soft_update(self.f_value, self.f_value_target, tau=self.tau)

    return {
        'embed_loss': loss,
        'reward_loss': tf.reduce_mean(reward_loss),
        'value_loss': tf.reduce_mean(value_loss),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, rewards, discounts, next_states = next(replay_buffer_iter)
    return self.fit(states, actions, rewards, discounts, next_states)

  def get_input_state_dim(self):
    return self.embedder.embedding_dim

