# -*- coding: utf-8 -*-
import tensorflow as tf

from capsule_masked import Capsule


def build_model(input_data, input_size, sequence_length, slot_size, intent_size, intent_dim, layer_size, embed_dim,
                num_rnn=1, isTraining=True, iter_slot=2, iter_intent=2, re_routing=True):
    cell_fw_list = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(layer_size) for _ in range(num_rnn)])
    cell_bw_list = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(layer_size) for _ in range(num_rnn)])

    if isTraining == True:
        cell_fw_list = tf.contrib.rnn.DropoutWrapper(cell_fw_list, input_keep_prob=0.8,
                                                     output_keep_prob=0.8)
        cell_bw_list = tf.contrib.rnn.DropoutWrapper(cell_bw_list, input_keep_prob=0.8,
                                                     output_keep_prob=0.8)

    embedding = tf.get_variable('embedding', [input_size, embed_dim],
                                initializer=tf.contrib.layers.xavier_initializer())
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    with tf.variable_scope('slot_capsule'):
        H, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [cell_fw_list],
            [cell_bw_list],
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32)
        sc = Capsule(slot_size, layer_size, reuse=tf.AUTO_REUSE, iter_num=iter_slot, wrr_dim=(layer_size, intent_dim))
        slot_capsule, routing_weight, routing_logits = sc(H, sequence_length, re_routing=False)
    with tf.variable_scope('slot_proj'):
        slot_p = tf.reshape(routing_logits, [-1, slot_size])
    with tf.variable_scope('intent_capsule'):
        intent_capsule, intent_routing_weight, _ = Capsule(intent_size, intent_dim, reuse=tf.AUTO_REUSE,
                                                           iter_num=iter_intent)(slot_capsule, slot_size)
    with tf.variable_scope('intent_proj'):
        intent = intent_capsule
    outputs = [slot_p, intent, routing_weight, intent_routing_weight]
    if re_routing:
        pred_intent_index_onehot = tf.one_hot(tf.argmax(tf.norm(intent_capsule, axis=-1), axis=-1), intent_size)
        pred_intent_index_onehot = tf.tile(tf.expand_dims(pred_intent_index_onehot, 2),
                                           [1, 1, tf.shape(intent_capsule)[2]])
        intent_capsule_max = tf.reduce_sum(tf.multiply(intent_capsule, tf.cast(pred_intent_index_onehot, tf.float32)),
                                           axis=1,
                                           keepdims=False)
        caps_ihat = tf.expand_dims(tf.expand_dims(intent_capsule_max, 1), 3)
        with tf.variable_scope('slot_capsule', reuse=True):
            slot_capsule_new, routing_weight_new, routing_logits_new = sc(H, sequence_length, caps_ihat=caps_ihat,
                                                                          re_routing=True)
        with tf.variable_scope('slot_proj', reuse=True):
            slot_p_new = tf.reshape(routing_logits_new, [-1, slot_size])
        outputs = [slot_p_new, intent, routing_weight_new, intent_routing_weight]
    return outputs
