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

    #dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.labels_dir, args.channels, seed=args.seed)
    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')
    #print(dir(data_generator))
    #print(data_generator.__len__())
    #input()

    for i, model in enumerate(models):
        print(f'Evaluating {weights[i]} model')
        if args.multi_gpu:
            model = multi_gpu_model(model. len(gpus))

        loss = make_loss(args.loss_function)
        model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

        metrics = {args.loss_function: [],
                   'binary_crossentropy': [],
                   'hard_dice_coef_ch1': [],
                   'hard_dice_coef': []}
        loop_stop = data_generator.__len__()

        counter = -1
        data_gen_len = data_generator.__len__()
        pred_mc = []
        pred_std_mc = np.zeros((data_gen_len, *data_generator.get_output_shape()[:2], args.out_channels))
        entropy_mc = np.zeros((data_gen_len, *data_generator.get_output_shape()[:2], args.out_channels))
        labels = []
        for x, y in tqdm(data_generator):
            counter += 1
            if counter >= loop_stop:
                break
            # old_predicts_x = []
            x_repeated = np.repeat(x, predictions_repetition, axis=0)
            predicts_x_repeated = model.predict(x_repeated, verbose=0)
            predicts_x = np.asarray([predicts_x_repeated[j*predictions_repetition:(j+1)*predictions_repetition, ...] for j in range(x.shape[0])])

            predicts_x_ext = np.repeat(predicts_x[..., np.newaxis], 2, axis=-1)
            predicts_x_ext[..., 1] = 1 - predicts_x_ext[..., 1]
            pred_mc.append(tf.math.reduce_mean(predicts_x_ext, axis=1)[0])
            pred_std_mc[counter] = np.sqrt(np.sum(np.var(predicts_x_ext, axis=1), axis=-1))
            entropy_mc[counter] = -np.sum(pred_mc[counter] * np.log2(pred_mc[counter] + 1E-14), axis=-1)  # Numerical Stability

            # for _ in range(predictions_repetition):
            #     old_predicts_x.append(model.predict(x, verbose=0))
            # old_mean_predicts = tf.math.reduce_mean(np.asarray(old_predicts_x), axis=0)
            mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=1)
            metrics[args.loss_function].append(loss(y, mean_predicts).numpy())
            metrics['binary_crossentropy'].append(binary_crossentropy(y, mean_predicts).numpy())
            metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, mean_predicts).numpy())
            metrics['hard_dice_coef'].append(hard_dice_coef(y, mean_predicts).numpy())

            # metrics[args.loss_function].append(loss(y, old_mean_predicts).numpy())
            # metrics['binary_crossentropy'].append(binary_crossentropy(y, old_mean_predicts).numpy())
            # metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, old_mean_predicts).numpy())
            # metrics['hard_dice_coef'].append(hard_dice_coef(y, old_mean_predicts).numpy())

            # print(loss(y, old_mean_predicts).numpy(), loss(y, mean_predicts).numpy())
            # print(binary_crossentropy(y, old_mean_predicts).numpy(), binary_crossentropy(y, mean_predicts).numpy())
            # print(hard_dice_coef_ch1(y, old_mean_predicts).numpy(), hard_dice_coef_ch1(y, mean_predicts).numpy())
            # print(hard_dice_coef(y, old_mean_predicts).numpy(), hard_dice_coef(y, mean_predicts).numpy())

            labels.append(*[y[i] for i in range(y.shape[0])])

            del x, y, predicts_x, mean_predicts
            gc.collect()

        mean_x_predicts = [p[..., 0] for p in pred_mc]
        new_loss_value, new_bce_value, new_hdc1_value, new_hdc_value = Mean()(list(map(lambda a: loss(a[0], a[1]), zip(labels, mean_x_predicts)))), \
                                                       Mean()(list(map(lambda a: binary_crossentropy(a[0], a[1]), zip(labels, mean_x_predicts)))), \
                                                       Mean()(list(map(lambda a: hard_dice_coef_ch1(a[0], a[1]), zip(labels, mean_x_predicts)))), \
                                                       Mean()(list(map(lambda a: hard_dice_coef(a[0], a[1]), zip(labels, mean_x_predicts))))

        #predicts, labels = tf.convert_to_tensor(np.asarray(predicts)), tf.convert_to_tensor(np.asarray(labels))
        loss_value, bce_value, hdc1_value, hdc_value = Mean()(metrics[args.loss_function]), \
                                                       Mean()(metrics['binary_crossentropy']), \
                                                       Mean()(metrics['hard_dice_coef_ch1']), \
                                                       Mean()(metrics['hard_dice_coef'])

        # loss_var, bce_var, hdc1_var, hdc_var = np.std(metrics[args.loss_function]), \
        #                                                np.std(metrics['binary_crossentropy']), \
        #                                                np.std(metrics['hard_dice_coef_ch1']), \
        #                                                np.std(metrics['hard_dice_coef'])

        print(f'Performed {predictions_repetition} repetitions per sample')
        print(f'{weights[i]} evaluation results:')
        # print(list(zip([args.loss_function, 'binary_crossentropy', 'hard_dice_coef_ch1', 'hard_dice_coef'], test_loss)))
        print(f'{args.loss_function}: {loss_value:.4f}, '
              f'binary_crossentropy: {bce_value:.4f}, '
              f'hard_dice_coef_ch1: {hdc1_value:.4f}, '
              f'hard_dice_coef: {hdc_value:.4f}')
        print(f'new_{args.loss_function}: {new_loss_value:.4f}, '
              f'new_binary_crossentropy: {new_bce_value:.4f}, '
              f'new_hard_dice_coef_ch1: {new_hdc1_value:.4f}, '
              f'new_hard_dice_coef: {new_hdc_value:.4f}')
        print('variances estimation')
        # print(f'{args.loss_function}: {loss_var:.4f}, '
        #       f'binary_crossentropy: {bce_var:.4f}, '
        #       f'hard_dice_coef_ch1: {hdc1_var:.4f}, '
        #       f'hard_dice_coef: {hdc_var:.4f}')


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
