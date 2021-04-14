import tensorflow as tf
import tensorflow.keras.backend as K


def brier_score(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    return K.mean(K.pow(y_pred_f - y_true_f, 2))


def actual_accuracy_and_confidence(y_true, y_pred):
    acc = K.cast(y_true[..., 0] == K.round(y_pred[..., 0]), dtype='float32')
    conf = y_pred[..., 0]
    return acc, conf


def entropy(y_pred):
    return y_pred * tf.math.log(y_pred + 1e-10)
