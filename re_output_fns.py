import tensorflow as tf
import re_nn_utils as nn_utils


def pos_softmax_classifier(inputs, targets, num_labels, tokens_to_keep):
    with tf.name_scope('joint_softmax_classifier'):
        # todo pass this as initial proj dim (which is optional)
        projection_dim = 200

        with tf.variable_scope('MLP'):
            mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=0.9, n_splits=1)
        with tf.variable_scope('Classifier'):
            logits = nn_utils.MLP(mlp, num_labels, keep_prob=0.9, n_splits=1)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

        cross_entropy *= tokens_to_keep
        loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)

        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        output = {
            'loss': loss,
            'predictions': predictions,
            'scores': logits,
            'probabilities': tf.nn.softmax(logits, -1)
        }

        return output


def parse_bilinear(inputs, targets, tokens_to_keep):
    class_mlp_size = 100
    attn_mlp_size = 500

    with tf.variable_scope('parse_bilinear'):
        with tf.variable_scope('MLP'):
            dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2, keep_prob=0.9)
            dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
            head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

        with tf.variable_scope('Arcs'):
            arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, 0.9)

        num_tokens = tf.reduce_sum(tokens_to_keep)

        predictions = tf.argmax(arc_logits, -1)
        probabilities = tf.nn.softmax(arc_logits)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
        loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / num_tokens

        output = {
            'loss': loss,
            'predictions': predictions,
            'probabilities': probabilities,
            'scores': arc_logits,
            'dep_rel_mlp': dep_rel_mlp,
            'head_rel_mlp': head_rel_mlp
        }

    return output


def conditional_bilinear(targets, num_labels, tokens_to_keep, dep_rel_mlp, head_rel_mlp, parse_preds_train):
    parse_preds = parse_preds_train
    with tf.variable_scope('conditional_bilin'):
        logits, _ = nn_utils.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_labels,
                                                             parse_preds, 0.9)

    predictions = tf.argmax(logits, -1)
    probabilities = tf.nn.softmax(logits)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

    n_tokens = tf.reduce_sum(tokens_to_keep)
    loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / n_tokens

    output = {
        'loss': loss,
        'scores': logits,
        'predictions': predictions,
        'probabilities': probabilities
    }

    return output


def ner_softmax_classifier(inputs, targets, num_labels, tokens_to_keep):
    with tf.name_scope('ner_softmax_classifier'):
        # todo pass this as initial proj dim (which is optional)
        projection_dim = 200

        with tf.variable_scope('NER'):
            mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=0.9, n_splits=1)
        with tf.variable_scope('NerClassifier'):
            logits = nn_utils.MLP(mlp, num_labels, keep_prob=0.9, n_splits=1)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

        cross_entropy *= tokens_to_keep
        loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)

        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        output = {
            'loss': loss,
            'predictions': predictions,
            'scores': logits,
            'probabilities': tf.nn.softmax(logits, -1)
        }

        return output

