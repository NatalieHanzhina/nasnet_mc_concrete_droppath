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
        loop_stop = data_generator.__len__()
        counter = -1
        pred_mc = []

        masks_unique_pixels = {}
        masks_unique_pixels_rate = {}
        masks_unique_pixels_keys = set()
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

            msk_unique_pixels = np.unique(y[:, :, :, 0], return_counts=True)
            to_add = {msk_unique_pixels[0][i]: msk_unique_pixels[1][i] for i in range(msk_unique_pixels[0].shape[0])}
            to_add['total'] = np.sum(msk_unique_pixels[1])
            for k in to_add.keys():
                if k not in masks_unique_pixels.keys():
                    masks_unique_pixels[k] = []
                masks_unique_pixels[k].append(to_add[k])

            to_add_ratios = to_add.copy()

            for k in to_add_ratios:
                if k == 'total':
                    continue
                if k not in masks_unique_pixels_rate.keys():
                    masks_unique_pixels_rate[k] = []
                masks_unique_pixels_rate[k].append(to_add_ratios[k] / to_add_ratios['total'])

            for k in to_add.keys():
                masks_unique_pixels_keys.add(k)

            del x, y, predicts_x, mean_predicts
            gc.collect()

        mean_masks_unique_pixels1 = {k: np.sum(masks_unique_pixels[k])/np.sum(masks_unique_pixels['total']) for k in masks_unique_pixels.keys()}
        #   Incorrect way! mean_masks_unique_pixels2 = {k: np.mean(masks_unique_pixels_rate[k]) for k in masks_unique_pixels_rate.keys()}

        print(f'mean masks unique pixels1:')
        for k in mean_masks_unique_pixels1:
            if k != 'total':
                print(f'\t{k}: {mean_masks_unique_pixels1[k]:.4f}')

        # print(f'\nmean masks unique pixels2:')
        # for k in mean_masks_unique_pixels2:
        #     print(f'\t{k}: {mean_masks_unique_pixels2[k]:.4f}')

        # print('\n\n'
        #       f'sensitivity2: {sensitivity2:.4f}\n'
        #       f'specificity2: {specificity2:.4f}\n'
        #       f'accuracy2: {accuracy2:.4f}\n'
        #       f'LR+2: {positive_likelihood_ratio2:.4f}\n'
        #       f'LR-2: {negative_likelihood_ratio2:.4f}\n'
        #       f'PPV2: {positive_predictive_value2:.4f}\n'
        #       f'NPV2: {negative_predictive_value2:.4f}\n')


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
