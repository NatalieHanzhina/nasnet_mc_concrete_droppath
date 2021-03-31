import tensorflow.keras.backend as K
import tensorflow as tf


def brier_score(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    return K.mean(K.pow(y_pred_f - y_true_f, 2))


def actual_accuracy_and_confidence(y_true, y_pred):
    acc = K.mean(y_true[..., 0] == K.round(y_pred[..., 0]), axis=list(range(1, len(y_true.shape)-1)))
    conf = K.mean(y_pred[..., 0], axis=list(range(1, len(y_true.shape)-1)))
    return acc, conf


def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def entropy(y_pred, sum_axis=None):
    if sum_axis is None:
        sum_axis = list(range(1, len(y_pred.shape)))
    return tf.reduce_sum(y_pred * tf.math.log(y_pred) / tf.math.log(2), axis=sum_axis)
