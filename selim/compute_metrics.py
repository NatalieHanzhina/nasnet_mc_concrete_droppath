import gc
import os

import numpy as np
from datasets.dsb_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef
from models.model_factory import make_model
from params import args
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(1)
import random

random.seed(1)
import tensorflow as tf

tf.random.set_seed(1)
import timeit
from tensorflow.keras.metrics import Mean
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import RMSprop

#test_pred = os.path.join(args.out_root_dir, args.out_masks_folder)


def main():
    all_ids = []
    all_images = []
    all_masks = []

    OUT_CHANNELS = args.out_channels
    TF_FORCE_GPU_ALLOW_GROWTH = True


    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



    predictions_repetition = 20
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
                               dp_p=args.dropout_rate)
            print("Building model {} from weights {} ".format(args.network, w))
            model.load_weights(w)
        models.append(model)

    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')

    for i, model in enumerate(models):
        print(f'Working with {weights[i]} model')
        if args.multi_gpu:
            model = multi_gpu_model(model. len(gpus))

        loss = make_loss(args.loss_function)
        model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

        metrics = {args.loss_function: [],
                   'TP': [],
                   'FP': [],
                   'TN': [],
                   'FN': []}

        loop_stop = data_generator.__len__()
        loop_stop = 5
        counter = -1
        pred_mc = []
        labels = []
        for x, y in tqdm(data_generator):
            counter += 1
            if counter >= loop_stop:
                break
            x_repeated = np.repeat(x, predictions_repetition, axis=0)
            predicts_x_repeated = model.predict(x_repeated, verbose=0)
            predicts_x = np.asarray(
                [predicts_x_repeated[j * predictions_repetition:(j + 1) * predictions_repetition, ...] for j in
                 range(x.shape[0])])

            predicts_x_ext = np.repeat(predicts_x[..., np.newaxis], 2, axis=-1)
            predicts_x_ext[..., 1] = 1 - predicts_x_ext[..., 1]
            pred_mc.append(tf.math.reduce_mean(predicts_x_ext, axis=1)[0])

            mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=1)
            metrics[args.loss_function].append(loss(y, mean_predicts).numpy())
            metrics['TP'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 1) & (y[:, :, :, 0] > 0)))
            metrics['FP'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 1) & (y[:, :, :, 0] == 0)))
            metrics['TN'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 0) & (y[:, :, :, 0] == 0)))
            metrics['FN'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 0) & (y[:, :, :, 0] > 0)))
            # metrics[args.loss_function].append(loss(y, old_mean_predicts).numpy())
            # metrics['binary_crossentropy'].append(binary_crossentropy(y, old_mean_predicts).numpy())
            # metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, old_mean_predicts).numpy())
            # metrics['hard_dice_coef'].append(hard_dice_coef(y, old_mean_predicts).numpy())
            assert metrics['TP'][-1] + metrics['FP'][-1] + metrics['TN'][-1] + metrics['FN'][-1] == y.shape[0]*y.shape[1]*y.shape[2]

            # print(loss(y, old_mean_predicts).numpy(), loss(y, mean_predicts).numpy())
            # print(binary_crossentropy(y, old_mean_predicts).numpy(), binary_crossentropy(y, mean_predicts).numpy())
            # print(hard_dice_coef_ch1(y, old_mean_predicts).numpy(), hard_dice_coef_ch1(y, mean_predicts).numpy())
            # print(hard_dice_coef(y, old_mean_predicts).numpy(), hard_dice_coef(y, mean_predicts).numpy())

            labels.append(*[y[i] for i in range(y.shape[0])])

            del x, y, predicts_x, mean_predicts
            gc.collect()

        loss_value = Mean()(metrics[args.loss_function])

        ld = 1e-8

        def compute_positive_likelihood_ratio(sensitivity, specificity):
            return sensitivity / (1 - specificity+ld)
        
        def compute_negative_likelihood_ratio(sensitivity, specificity):
            return (1 - sensitivity) / (specificity+ld)

        sensitivity1 = np.mean([metrics['TP'][i]/(metrics['TP'][i]+metrics['FN'][i]+ld) for i in range(counter)])
        specificity1 = np.mean([metrics['TN'][i]/(metrics['TN'][i]+metrics['FP'][i]+ld) for i in range(counter)])
        accuracy1 = np.mean([(metrics['TP'][i] + metrics['TN'][i]) / (
                    metrics['TP'][i] + metrics['FP'][i] + metrics['TN'][i] + metrics['FN'][i]+ld) for i in range(counter)])
        positive_likelihood_ratio1 = np.mean(
            [compute_positive_likelihood_ratio(metrics['TP'][i] / (metrics['TP'][i] + metrics['FN'][i]+ld),
                                               metrics['TN'][i] / (metrics['TN'][i] + metrics['FP'][i]+ld)) for i in
            range(counter)])

        negative_likelihood_ratio1 = np.mean(
            [compute_negative_likelihood_ratio(metrics['TP'][i] / (metrics['TP'][i] + metrics['FN'][i]+ld),
                                              metrics['TN'][i] / (metrics['TN'] + metrics['FP'][i]+ld)) for i in
            range(counter)])

        positive_predictive_value1 = np.mean([metrics['TP'][i] / (metrics['TP'][i] + metrics['FP'][i]+ld) for i in range(counter)])
        negative_predictive_value1 = np.mean([metrics['TN'][i] / (metrics['TN'][i] + metrics['FN'][i]+ld) for i in range(counter)])


        print(f'Performed {predictions_repetition} repetitions per sample')
        print(f'{weights[i]} evaluation results:')
        print(f'{args.loss_function}: {loss_value:.4f}\n'
              f'sensitivity: {sensitivity1:.4f}\n'
              f'specificity: {specificity1:.4f}\n'
              f'accuracy1: {accuracy1:.4f}\n'
              f'LR+: {positive_likelihood_ratio1:.4f}\n'
              f'LR-: {negative_likelihood_ratio1:.4f}\n'
              f'PPV: {positive_predictive_value1:.4f}\n'
              f'NPV: {negative_predictive_value1:.4f}\n')
        # print(f'new_{args.loss_function}: {new_loss_value:.4f}, '
        #       f'new_binary_crossentropy: {new_bce_value:.4f}, '
        #       f'new_hard_dice_coef_ch1: {new_hdc1_value:.4f}, '
        #       f'new_hard_dice_coef: {new_hdc_value:.4f}')


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
        # if l.name != d_l.name: # incorrect names
        if 'dropout' in l.name and 'dropout' not in d_l.name:
            continue

        j += 1
        for (w, d_w) in zip(l.weights, d_l.weights):
            w.assign(d_w)

    assert j == len(donor_model.layers)
    return model


if __name__ == '__main__':
    main()