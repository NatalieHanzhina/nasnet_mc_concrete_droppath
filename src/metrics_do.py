import tensorflow.keras.backend as K


def brier_score(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    return K.mean(K.pow(y_pred_f - y_true_f, 2))


def actual_accuracy_and_confidence(y_true, y_pred):
    acc = K.mean(y_true == y_pred, axis=list(range(1, len(y_true.shape))))
    conf = K.mean(y_pred, axis=list(range(1, len(y_true.shape))))
    return acc, conf


def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)