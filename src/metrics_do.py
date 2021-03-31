import tensorflow.keras.backend as K
import tensorflow as tf


def brier_score(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    return K.mean(K.pow(y_pred_f - y_true_f, 2))


def actual_accuracy_and_confidence(y_true, y_pred):
    acc = K.cast(y_true[..., 0] == K.round(y_pred[..., 0]), dtype='float32')
    conf = y_pred[..., 0]
    return acc, conf


def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def entropy(y_pred):
    #tf.print('entropy_result_shape:', (y_pred * tf.math.log(y_pred)).shape)
    return y_pred * tf.math.log(y_pred + 1e-10)
