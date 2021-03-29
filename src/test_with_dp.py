import gc
import os

import numpy as np

from datasets.dsb_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef
from metrics_do import brier_score, actual_accuracy_and_confidence
from models.model_factory import make_model
from params import args

np.random.seed(1)
import random

random.seed(1)
import tensorflow as tf
import tensorflow_probability as tfp

tf.random.set_seed(1)
import timeit
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import RMSprop

#test_pred = os.path.join(args.out_root_dir, args.out_masks_folder)


def main():
    all_ids = []
    all_images = []
    all_masks = []

    OUT_CHANNELS = args.out_channels

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    predictions_repetition = args.times_sample_per_test
    model_without_dropout = False
    t0 = timeit.default_timer()

    weights = [os.path.join(args.models_dir, m) for m in args.models]
    models = []
    for w in weights:
        if model_without_dropout:
            model = load_model_weights(w)
        else:
            model = make_model(args.network,
                               (None, None, args.channels),
                               pretrained_weights=args.pretrained_weights,
                               do_p=args.dropout_rate,
                               resize_size=(args.resize_size, args.resize_size))
            print("Building model {} from weights {} ".format(args.network, w))
            model.load_weights(w)
        models.append(model)

    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')

    for i, model in enumerate(models):
        print(f'Evaluating {weights[i]} model')
        loss = make_loss(args.loss_function)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

        metrics = {args.loss_function: [],
                   'binary_crossentropy': [],
                   'hard_dice_coef_ch1': [],
                   'hard_dice_coef': [],
                   'brier_score': [],
                   'tf_brier_score': [],
                   'expected_calibration_error': []
                   }
        loop_stop = data_generator.__len__()

        counter = -1
        data_gen_len = data_generator.__len__()
        pred_mc = []
        pred_std_mc = np.zeros((data_gen_len, *data_generator.get_output_shape()[:2], args.out_channels))
        entropy_mc = np.zeros((data_gen_len, *data_generator.get_output_shape()[:2], args.out_channels))
        prog_bar = tf.keras.utils.Progbar(data_gen_len)
        for x, y in data_generator:
            counter += 1
            if counter >= loop_stop:
                break
            x_repeated = np.repeat(x, predictions_repetition, axis=0)
            predicts_x_repeated = model.predict(x_repeated, verbose=0)
            predicts_x = np.asarray([predicts_x_repeated[j*predictions_repetition:(j+1)*predictions_repetition, ...] for j in range(x.shape[0])])

            # predicts_x_ext = np.repeat(predicts_x[..., np.newaxis], 2, axis=-1)
            # predicts_x_ext[..., 1] = 1 - predicts_x_ext[..., 1]
            # pred_mc.append(tf.math.reduce_mean(predicts_x_ext, axis=1)[0])
            # pred_std_mc[counter] = np.sqrt(np.sum(np.var(predicts_x_ext, axis=1), axis=-1))
            # entropy_mc[counter] = -np.sum(pred_mc[counter] * np.log2(pred_mc[counter] + 1E-14), axis=-1)  # Numerical Stability

            mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=1)
            metrics[args.loss_function].append(loss(y, mean_predicts).numpy())
            metrics['binary_crossentropy'].append(binary_crossentropy(y, mean_predicts).numpy())
            metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, mean_predicts).numpy())
            metrics['hard_dice_coef'].append(hard_dice_coef(y, mean_predicts).numpy())
            metrics['brier_score'].append(brier_score(y, mean_predicts).numpy())
            metrics['tf_brier_score'].append(tfp.stats.brier_score(y.astype(np.int32)[..., 0], mean_predicts[..., 0]).numpy())
            metrics['expected_calibration_error'].append(actual_accuracy_and_confidence(y.astype(np.int32), mean_predicts))

            exclude_metrics = ['tf_brier_score', 'expected_calibration_error']
            # [(k,v[-1]) for k,v in metrics.items() if k not in exclude_metrics]
            prog_bar.update(counter+1, [(k, round(v[-1], 4)) for k,v in metrics.items() if k not in exclude_metrics])

            del x, y, predicts_x, mean_predicts
            gc.collect()

        loss_value, bce_value, hdc1_value, hdc_value, brier_score_value, tf_brier_score_value = \
            Mean()(metrics[args.loss_function]), \
            Mean()(metrics['binary_crossentropy']), \
            Mean()(metrics['hard_dice_coef_ch1']), \
            Mean()(metrics['hard_dice_coef']), \
            Mean()(metrics['brier_score']), \
            Mean()(metrics['tf_brier_score'])

        m = 20
        groups = data_gen_len // m
        eces = []
        j = 0
        for j in range(1, groups):
            accs, probs = zip(*metrics['expected_calibration_error'][(j-1)*m:j*m])
            eces.append(m/data_gen_len*tf.abs(tf.keras.backend.mean(accs) - tf.keras.backend.mean(probs)))
            pass
        accs, probs = zip(*metrics['expected_calibration_error'][(j) * m:])
        eces.append((data_gen_len % m) / data_gen_len * tf.abs(tf.reduce_mean(accs) - tf.reduce_mean(probs)))
        ece_value = tf.keras.backend.sum(eces)

        print(f'Performed {predictions_repetition} repetitions per sample')
        print(f'Dropout rate: {args.dropout_rate}')
        print(f'{weights[i]} evaluation results:')
        print(f'{args.loss_function}: {loss_value:.4f}, '
              f'binary_crossentropy: {bce_value:.4f}, '
              f'hard_dice_coef_ch1: {hdc1_value:.4f}, '
              f'hard_dice_coef: {hdc_value:.4f}')
        print('Monte-Calro estimation')
        print(f'brier_score: {brier_score_value:.4f}, '
              f'tf_brier_score: {tf_brier_score_value:.4f}',
              f'exp_calibration_error: {ece_value:.4f}')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    exit(0)


def load_model_weights(w):
    model = make_model(args.network, (None, None, args.channels), pretrained_weights=args.pretrained_weights)
    donor_model = make_model(args.network[:-3], (None, None, args.channels), pretrained_weights=args.pretrained_weights)
    print("Building dropout model {} from weights without dropout {} ".format(args.network, w))
    donor_model.load_weights(w)

    j = 0
    for i, l in enumerate(model.layers):
        if j >= len(donor_model.layers):
            break
        d_l = donor_model.layers[j]
        if 'dropout' in l.name and 'dropout' not in d_l.name:
            continue

        j += 1
        for (w, d_w) in zip(l.weights, d_l.weights):
            w.assign(d_w)

    assert j == len(donor_model.layers)
    return model


if __name__ == '__main__':
    main()
