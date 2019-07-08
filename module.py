# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def _softmax_with_mask(logits, lens, axis=-1):
    """Helper function for softmax on variable-length sequences.
        Args:
            logits: The logits before softmax. Shape is [batch, type_num, class_num]
            lens: The length of the sequence. Shape is [batch, type_num].
            axis: The axis to apply softmax operator on.
        Returns:
             A tensor with softmax-ed values. Same shape as logits.
    """
    exp_logits = tf.exp(logits)
    mask = tf.sequence_mask(lens, maxlen=tf.shape(logits)[axis], dtype=tf.float32)
    masked_exp_logits = tf.multiply(exp_logits, mask)
    masked_exp_logits_sum = tf.reduce_sum(masked_exp_logits, axis)
    return tf.clip_by_value(tf.div(masked_exp_logits, tf.expand_dims(masked_exp_logits_sum, axis)), 1e-37, 1e+37)


def _squash(input_tensor):
    """Applies norm nonlinearity (squash) to a capsule layer.
        Args:
            input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
              fully connected capsule layer or
              [batch, num_channels, num_atoms, height, width] for a convolutional
              capsule layer.
        Returns:
            A tensor with same shape as input (rank 3) for output of this layer.
    """
    with tf.name_scope('norm_non_linearity'):
        norm = tf.norm(input_tensor, axis=2, keep_dims=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


def _leaky_routing(logits, output_dim):
    """Adds extra dimmension to routing logits.
    This enables active capsules to be routed to the extra dim if they are not a
    good fit for any of the capsules in layer above.
    Args:
      logits: The original logits. shape is
        [input_capsule_num, output_capsule_num] if fully connected. Otherwise, it
        has two more dimmensions.
      output_dim: The number of units in the second dimmension of logits.
    Returns:
      Routing probabilities for each pair of capsules. Same shape as logits.
    """
    leak = tf.zeros_like(logits, optimize=True)
    leak = tf.reduce_sum(leak, axis=2, keep_dims=True)
    leaky_logits = tf.concat([leak, logits], axis=2)
    leaky_routing = tf.nn.softmax(leaky_logits, dim=2)
    return tf.split(leaky_routing, [1, output_dim], 2)[1]


def _update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                    num_routing=3, leaky=True):
    """Sums over scaled votes and applies squash to compute the activations.
    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.
    Args:
      votes: tensor, The transformed outputs of the layer below.
      biases: tensor, Bias variable.
      logit_shape: tensor, shape of the logit to be initialized.
      num_dims: scalar, number of dimmensions in votes. For fully connected
        capsule it is 4, for convolutional 6.
      input_dim: scalar, number of capsules in the input layer.
      output_dim: scalar, number of capsules in the output layer.
      num_routing: scalar, Number of routing iterations.
      leaky: boolean, if set use leaky routing.
    Returns:
      The activation tensor of the output layer after num_routing iterations.
    """
    votes_t_shape = [3, 0, 1, 2]
    for i in range(num_dims - 4):
        votes_t_shape += [i + 4]
    r_t_shape = [1, 2, 3, 0]
    for i in range(num_dims - 4):
        r_t_shape += [i + 4]
    votes_trans = tf.transpose(votes, votes_t_shape)

    def _body(i, logits, activations, routes):
        """Routing while loop."""
        if leaky:
            route = _leaky_routing(logits, output_dim)
        else:
            route = tf.nn.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        routes = routes.write(i, route)
        # distances: [batch, input_dim, output_dim]
        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations, routes)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    routes = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations, routes = tf.while_loop(
        lambda i, logits, activations, routes: i < num_routing,
        _body,
        loop_vars=[i, logits, activations, routes],
        swap_memory=True)

    return activations.read(num_routing - 1), logits, routes.read(num_routing - 1)


class Capsule:
    def __init__(self, input_dim, input_atoms, output_dim, output_atoms, layer_name):
        self.input_dim = input_dim
        self.input_atoms = input_atoms
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        with tf.variable_scope(layer_name):
            self.weights = tf.get_variable(name='w',
                                           shape=[1, input_dim, input_atoms, output_dim * output_atoms],
                                           dtype=tf.float32)
            self.biases = tf.get_variable(name='b', shape=[output_dim, output_atoms], dtype=tf.float32,
                                          initializer=tf.zeros_initializer())

    def vote_and_route(self, input_tensor, leaky=False):
        with tf.name_scope('Wx_plus_b'):
            input_tiled = tf.tile(tf.expand_dims(input_tensor, -1),
                                  [1, 1, 1, self.output_dim * self.output_atoms])
            votes = tf.reduce_sum(input_tiled * self.weights, axis=2)
            votes_reshaped = tf.reshape(votes,
                                        [-1, self.input_dim, self.output_dim, self.output_atoms])
        with tf.name_scope('routing'):
            input_shape = tf.shape(input_tensor)
            logit_shape = tf.stack([input_shape[0], self.input_dim, self.output_dim])
            activations, weights_c, route = _update_routing(
                votes=votes_reshaped,
                biases=self.biases,
                logit_shape=logit_shape,
                num_dims=4,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                leaky=leaky,
                num_routing=3)
        return activations, weights_c, route
