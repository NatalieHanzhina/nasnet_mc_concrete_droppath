import os
import random
import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import RMSprop

from datasets.dsb_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef_combined, hard_dice_coef
from metrics_do import actual_accuracy_and_confidence, brier_score, entropy
from models.model_factory import make_model
from params import args

# import tensorflow_probability as tfp

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)
#test_pred = os.path.join(args.out_root_dir, args.out_masks_folder)
test_pred = args.out_root_dir
print('test predictions folder:', end=' ')
print(test_pred)

def main():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    t0 = timeit.default_timer()

    weights = [os.path.join(args.models_dir, m) for m in args.models]
    models = []
    for w in weights:
        model = make_model(args.network,
                           (None, None, args.channels),
                           pretrained_weights=args.pretrained_weights,
                           do_p=args.dropout_rate,
                           resize_size=(args.resize_size, args.resize_size))
        print("Building model {} from weights {} ".format(args.network, w))
        print(f'Using dropout rate {args.dropout_rate}')
        model.load_weights(w)
        models.append(model)

    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_ensemble_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')

    loop_stop = data_generator.__len__()
    data_gen_len = data_generator.__len__()
    models_predicts = []
    samples_paths = []
    loss = make_loss(args.loss_function)
    for i, model in enumerate(models):
        print(f'Evaluating {weights[i]} model')
        #loss = make_loss(args.loss_function)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

        #models_predicts.append(model.predict(data_generator))
        model_predicts = None
        samples_path = None
        prog_bar = tf.keras.utils.Progbar(data_gen_len)
        for s_i, ((x, y), sample_path) in enumerate(data_generator):
            #if s_i >= 10:
            if s_i >= loop_stop:
                break
            m_pred = model.predict(x)
            model_predicts = m_pred if model_predicts is None else np.concatenate((model_predicts, m_pred), axis=0)
            samples_path = sample_path if samples_path is None else np.concatenate((samples_path, sample_path), axis=0)
            prog_bar.update(s_i+1)
        models_predicts.append(model_predicts)
        samples_paths.append(samples_path)
    models_predicts= np.asarray(models_predicts)
    samples_paths = np.asarray(samples_paths)

    print('Saving model predicts')
    for i, model_predicts in enumerate(models_predicts):
        f_path = os.path.join(test_pred, os.path.split(weights[i])[-1][:-3]+'__predicts.npy')
        with open(f_path, 'wb') as pred_f:
            np.save(pred_f, models_predicts)
        print('\nsaved_to_file>>>>>'+f_path+'<<<<<\n')

        f_path = os.path.join(test_pred, os.path.split(weights[i])[-1][:-3] + '__samples_paths.npy')
        with open(f_path, 'wb') as samp_paths_f:
            np.save(samp_paths_f, samples_paths[i])
            #samp_paths_f.writelines(samples_paths[i])
    return

    # counter = -1
    # data_gen_len = data_generator.__len__()
    entropy_of_mean = []
    mean_entropy = []
    metrics = {args.loss_function: [],
               'binary_crossentropy': [],
               'hard_dice_coef_ch1': [],
               'hard_dice_coef': [],
               'hard_dice_coef_combined': [],
               'brier_score': [],
               'expected_calibration_error': []
               }
    for i, (x,y) in enumerate(data_generator):
        # counter += 1
        # if counter >= loop_stop:
        #     break

        ensemble_predict = np.asarray([models_predicts[j][i] for j in range(len(models_predicts))])
        mean_ens_predict = np.mean(ensemble_predict, axis=0)


        # x_repeated = np.repeat(x, predictions_repetition, axis=0)
        # predicts_x_repeated = model.predict(x_repeated, verbose=0)
        # predicts_x = np.asarray([predicts_x_repeated[j*predictions_repetition:(j+1)*predictions_repetition, ...] for j in range(x.shape[0])])

        # mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=1)
        metrics[args.loss_function].append(loss(y, mean_ens_predict).numpy())
        metrics['binary_crossentropy'].append(binary_crossentropy(y, mean_ens_predict).numpy())
        metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, mean_ens_predict).numpy())
        metrics['hard_dice_coef'].append(hard_dice_coef(y, mean_ens_predict).numpy())
        metrics['hard_dice_coef_combined'].append(hard_dice_coef_combined(y, mean_ens_predict).numpy())
        metrics['brier_score'].append(brier_score(y, mean_ens_predict).numpy())
        metrics['expected_calibration_error'].append(actual_accuracy_and_confidence(y.astype(np.int32), mean_ens_predict))

        mean_entropy.append(tf.reduce_mean(entropy(ensemble_predict[..., 0]), axis=1))
        #tf.print('m_e:',tf.shape(mean_entropy[-1]))
        entropy_of_mean.append(entropy(mean_ens_predict[..., 0]))
        #tf.print('e_o_m:',tf.shape(entropy_of_mean[-1]))


        #del x, y, ensemble_predict, mean_ens_predict
        #gc.collect()

    loss_value, bce_value, hdc1_value, hdc_value, hdcc_value, brier_score_value = \
        Mean()(metrics[args.loss_function]), \
        Mean()(metrics['binary_crossentropy']), \
        Mean()(metrics['hard_dice_coef_ch1']), \
        Mean()(metrics['hard_dice_coef']), \
        Mean()(metrics['hard_dice_coef_combined']), \
        Mean()(metrics['brier_score'])

    m = 10
    accs, probs = zip(*metrics['expected_calibration_error'])
    accs, probs = np.concatenate(np.asarray(accs), axis=0), np.concatenate(np.asarray(probs), axis=0)
    ece1_value = compute_ece1(accs, probs, m)
    correct_ece_value = compute_correct_ece(accs, probs, m)
    #tf.print(tf.convert_to_tensor(eces).shape)

    #tf.print(np.asarray(mean_entropy).shape, np.asarray(entropy_of_mean).shape)
    mean_entropy_subtr = np.mean(np.asarray(mean_entropy)-np.asarray(entropy_of_mean))
    #mean_entropy_subtr = tf.reduce_mean(mean_entropy-entropy_of_mean)

    print(f'Dropout rate: {args.dropout_rate}')
    print({"\n".join(weights)}, 'evaluation results:')
    print(f'{args.loss_function}: {loss_value:.4f}, '
          f'binary_crossentropy: {bce_value:.4f}, '
          f'hard_dice_coef_ch1: {hdc1_value:.4f}, '
          f'hard_dice_coef: {hdc_value:.4f}',
          f'hard_dice_coef_combined: {hdcc_value:.4f}')
    print('Monte-Calro estimation')
    print(f'brier_score: {brier_score_value:.4f}, '
          f'correct_exp_calibration_error: {correct_ece_value:.4f}',
          f'exp_calibration_error1: {ece1_value:.4f}',
          f'\nmean_entropy_subtr: {mean_entropy_subtr:.4f}')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    exit(0)


def compute_correct_ece(accs, probs, bins):
    pixel_wise_eces = []
    accs = accs.flatten()
    probs = probs.flatten()
    probs_min = np.min(probs)
    h_w_wise_bins_len = (np.max(probs)-probs_min) / bins
    for j in range(bins):
        # tf.print(tf.convert_to_tensor(accs).shape, tf.convert_to_tensor(probs).shape)
        if j == 0:
            include_flags = np.logical_and(probs >= probs_min + (h_w_wise_bins_len*j), probs <= probs_min + (h_w_wise_bins_len*(j+1)))
        else:
            include_flags = np.logical_and(probs > probs_min + (h_w_wise_bins_len*j), probs <= probs_min + (h_w_wise_bins_len*(j+1)))
        if np.sum(include_flags) == 0:
            continue
        included_accs = accs[include_flags]
        included_probs = probs[include_flags]
        mean_accuracy = included_accs.mean()
        mean_confidence = included_probs.mean()
        bin_ece = np.abs(mean_accuracy-mean_confidence)*np.sum(include_flags, axis=-1)
        pixel_wise_eces.append(bin_ece)
    pixel_wise_ece = np.sum(np.asarray(pixel_wise_eces), axis=0) / accs.shape[-1]
    return pixel_wise_ece.mean()


def compute_ece1(accs, probs, bins):
    pixel_wise_eces = []
    accs = np.transpose(accs, axes=(1, 2, 0))
    probs = np.transpose(probs, axes=(1, 2, 0))
    probs_mins = np.min(probs, axis=2)
    h_w_wise_bins_len = (np.max(probs, axis=2)-probs_mins) / bins
    for j in range(bins):
        # tf.print(tf.convert_to_tensor(accs).shape, tf.convert_to_tensor(probs).shape)
        if j == 0:
            include_flags = np.logical_and(probs >= probs_mins[..., np.newaxis]+(h_w_wise_bins_len*j)[..., np.newaxis], probs <= probs_mins[..., np.newaxis] + (h_w_wise_bins_len*(j+1))[..., np.newaxis])
        else:
            include_flags = np.logical_and(probs > probs_mins[..., np.newaxis] + (h_w_wise_bins_len*j)[..., np.newaxis], probs <= probs_mins[..., np.newaxis] + (h_w_wise_bins_len*(j+1))[..., np.newaxis])
        if np.sum(include_flags) == 0:
            continue
        masked_accs = np.ma.masked_where(include_flags, accs)
        masked_probs = np.ma.masked_where(include_flags, probs)
        mean_accuracy = masked_accs.mean(axis=-1)
        mean_confidence = masked_probs.mean(axis=-1)
        pixel_wise_ece = np.ma.abs(mean_accuracy-mean_confidence)*np.sum(include_flags, axis=-1)
        pixel_wise_eces.append(pixel_wise_ece)
    pixel_wise_ece = np.sum(np.asarray(pixel_wise_eces), axis=0) / accs.shape[-1]
    return pixel_wise_ece.mean()


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
