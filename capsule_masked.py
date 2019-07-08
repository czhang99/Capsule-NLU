# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base as base_layer

from TfUtils import mkMask

_EPSILON = 1e-9
_MIN_NUM = -np.Inf


class Capsule(base_layer.Layer):
    def __init__(self, out_caps_num, out_caps_dim, iter_num=3, wrr_dim=(1, 1), reuse=None):
        super(Capsule, self).__init__(_reuse=reuse)
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.iter_num = iter_num
        self.w_rr = tf.get_variable(name='w_rr', shape=(1, 1, wrr_dim[0], wrr_dim[1]))

    def call(self, in_caps, seqLen, caps_ihat=None, re_routing=False):
        caps_uhat = shared_routing_uhat(in_caps, self.out_caps_num, self.out_caps_dim, scope='rnn_caps_uhat')
        if not re_routing:
            V, S, C, B = masked_routing_iter(caps_uhat, seqLen, self.iter_num, caps_ihat, w_rr=None)
        else:
            V, S, C, B = masked_routing_iter(caps_uhat, seqLen, self.iter_num, caps_ihat, w_rr=self.w_rr)
        return V, C, B


def shared_routing_uhat(caps, out_caps_num, out_caps_dim, scope=None):
    '''

    Args:
        caps: # shape(b_sz, caps_num, caps_dim)
        out_caps_num: #number of output capsule
        out_caps_dim: #dimension of output capsule
    Returns:
        caps_uhat: shape(b_sz, caps_num, out_caps_num, out_caps_dim)
    '''
    b_sz = tf.shape(caps)[0]
    tstp = tf.shape(caps)[1]

    with tf.variable_scope(scope or 'shared_routing_uhat'):
        '''shape(b_sz, caps_num, out_caps_num*out_caps_dim)'''
        caps_uhat = tf.layers.dense(caps, out_caps_num * out_caps_dim, activation=tf.tanh)
        caps_uhat = tf.reshape(caps_uhat, shape=[b_sz, tstp, out_caps_num, out_caps_dim])
    return caps_uhat


def masked_routing_iter(caps_uhat, seqLen, iter_num, caps_ihat=None, w_rr=None):
    '''

    Args:
        caps_uhat:  shape(b_sz, tstp, out_caps_num, out_caps_dim)
        seqLen:     shape(b_sz)
        iter_num:   number of iteration

    Returns:
        V_ret:      #shape(b_sz, out_caps_num, out_caps_dim)
    '''
    assert iter_num > 0
    b_sz = tf.shape(caps_uhat)[0]
    tstp = tf.shape(caps_uhat)[1]
    out_caps_num = int(caps_uhat.get_shape()[2])
    seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)
    mask = mkMask(seqLen, tstp)  # shape(b_sz, tstp)
    floatmask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)  # shape(b_sz, tstp, 1)
    B = tf.zeros([b_sz, tstp, out_caps_num], dtype=tf.float32)
    C_list = list()
    for i in range(iter_num):
        B_logits = B
        C = tf.nn.softmax(B, axis=2)  # shape(b_sz, tstp, out_caps_num)
        C = tf.expand_dims(C * floatmask, axis=-1)  # shape(b_sz, tstp, out_caps_num, 1)
        weighted_uhat = C * caps_uhat  # shape(b_sz, tstp, out_caps_num, out_caps_dim)
        C_list.append(C)
        S = tf.reduce_sum(weighted_uhat, axis=1)  # shape(b_sz, out_caps_num, out_caps_dim)
        V = _squash(S, axes=[2])  # shape(b_sz, out_caps_num, out_caps_dim)
        V = tf.expand_dims(V, axis=1)  # shape(b_sz, 1, out_caps_num, out_caps_dim)
        if caps_ihat == None:
            B = tf.reduce_sum(caps_uhat * V, axis=-1) + B  # shape(b_sz, tstp, out_caps_num)
        else:
            B = tf.reduce_sum(caps_uhat * V, axis=-1) + 0.1 * tf.squeeze(
                tf.matmul(tf.matmul(caps_uhat, tf.tile(w_rr, [tf.shape(caps_uhat)[0], tf.shape(caps_uhat)[1], 1, 1])),
                          tf.tile(caps_ihat, [1, tf.shape(caps_uhat)[1], 1, 1])),
                axis=-1) + B  # shape(b_sz, tstp, out_caps_num)
    V_ret = tf.squeeze(V, axis=[1])  # shape(b_sz, out_caps_num, out_caps_dim)
    S_ret = S
    C_ret = tf.squeeze(tf.stack(C_list), axis=[4])
    return V_ret, S_ret, C_ret, B_logits


def margin_loss(y_true, y_pred):
    """
    :param y_true: [None, n_classes]
    :param y_pred: [None, n_classes]
    :return: a scalar loss value.
    """
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    assert_inf_L = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_inf(L))),
                             ['assert_inf_L', L], summarize=100)
    assert_nan_L = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(L))),
                             ['assert_nan_L', L], summarize=100)
    with tf.control_dependencies([assert_inf_L, assert_nan_L]):
        ret = tf.reduce_mean(tf.reduce_sum(L, axis=1))
    return ret


def _squash(in_caps, axes):
    '''
    Squashing function corresponding to Eq. 1
    Args:
        in_caps:  a tensor
        axes:     dimensions along which to apply squash

    Returns:
        vec_squashed:   squashed tensor

    '''
    vec_squared_norm = tf.reduce_sum(tf.square(in_caps), axis=axes, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + _EPSILON)
    vec_squashed = scalar_factor * in_caps  # element-wise
    return vec_squashed
