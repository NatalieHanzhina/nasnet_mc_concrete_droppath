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


    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



    predictions_repetition = 50
    model_without_dropout = False
    t0 = timeit.default_timer()

    weights = [os.path.join(args.models_dir, m) for m in args.models]
    models = []
    for w in weights:
        if model_without_dropout:
            model = load_model_weights(w)
        else:
            model = make_model(args.network, (None, None, args.channels), pretrained_weights=args.pretrained_weights)
            print("Building model {} from weights {} ".format(args.network, w))
            model.load_weights(w)
        models.append(model)
    #os.makedirs(test_pred, exist_ok=True)

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
        counter = 0
        for x, y in tqdm(data_generator):
            counter +=1
            if counter > loop_stop:
                break
            predicts_x = []
            for _ in range(predictions_repetition):
                aug_x, aug_y, *aug_params = augment_batch(x, y)
                for a in range(aug_x.shape[0]):
                    predicts_x.append([])
                    for b in range(aug_x.shape[1]):
                        predicts_x[a].append(model.predict(aug_x[a, b], verbose=0))

            mean_model_predicts()


            mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=0)
            metrics[args.loss_function].append(loss(y, mean_predicts))
            metrics['binary_crossentropy'].append(binary_crossentropy(y, mean_predicts))
            metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, mean_predicts))
            metrics['hard_dice_coef'].append(hard_dice_coef(y, mean_predicts))

            #print('\n', mean_predicts.shape)
            #print(np.asarray(predicts_x).shape, mean_predicts.shape, y.shape)
            # input()
            #print(f'{args.loss_function}: {loss(y, mean_predicts):.4f}, '
            #      f'binary_crossentropy: {binary_crossentropy(y, mean_predicts):.4f}, '
            #      f'hard_dice_coef_ch1: {hard_dice_coef_ch1(y, mean_predicts):.4f}, '
            #      f'hard_dice_coef: {hard_dice_coef(y, mean_predicts):.4f}')
            gc.collect()


        #predicts, labels = tf.convert_to_tensor(np.asarray(predicts)), tf.convert_to_tensor(np.asarray(labels))
        loss_value, bce_value, hdc1_value, hdc_value = Mean()(metrics[args.loss_function]), \
                                                       Mean()(metrics['binary_crossentropy']), \
                                                       Mean()(metrics['hard_dice_coef_ch1']), \
                                                       Mean()(metrics['hard_dice_coef'])
        print(f'Performed {predictions_repetition} repetitions per sample')
        print(f'{weights[i]} evaluation results:')
        # print(list(zip([args.loss_function, 'binary_crossentropy', 'hard_dice_coef_ch1', 'hard_dice_coef'], test_loss)))
        print(f'{args.loss_function}: {loss_value:.4f}, '
              f'binary_crossentropy: {bce_value:.4f}, '
              f'hard_dice_coef_ch1: {hdc1_value:.4f}, '
              f'hard_dice_coef: {hdc_value:.4f}')

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


def augment_batch(batch_x, batch_y):
    shift_ratio = 0.02

    aug_batch_x = []
    aug_batch_y = []
    flip_flags = (True, False)
    shifts_by_x = []
    shifts_by_y = []
    for flip_flag in flip_flags:

        cur_batch_x = np.asarray(
            [np.flip(batch_x[i], 0) for i in range(batch_x.shape[0])]) if flip_flag else batch_x
        cur_batch_y = np.asarray(
            [np.flip(batch_y[i], 0) for i in range(batch_y.shape[0])]) if flip_flag else batch_y

        shfts_by_x = []
        shfts_by_y = []
        for i in range(4):
            shift_by_x = np.random.randint(-batch_x.shape[1] * shift_ratio, batch_x.shape[1] * shift_ratio,
                                           size=(batch_x.shape[0]))
            shift_by_x = np.asarray(list(map(lambda x: (int(abs(x) - x), int(abs(x)+x)), shift_by_x)))
            shfts_by_x.append(shift_by_x)

            shift_by_y = np.random.randint(-batch_x.shape[2] * shift_ratio, batch_x.shape[2] * shift_ratio,
                                           size=(batch_x.shape[0]))
            shift_by_y = np.asarray(list(map(lambda y: (int(abs(y) - y), int(abs(y) + y)), shift_by_y)))
            shfts_by_y.append(shift_by_y)
        shifts_by_x.append(shfts_by_x)
        shifts_by_y.append(shfts_by_y)

        shifted_batch_x = []
        shifted_batch_y = []
        for i in range(4):
            shift_by_x = shfts_by_x[i]
            shift_by_y = shfts_by_y[i]
            shfd_batch_x_lst = []
            for i in range(cur_batch_x.shape[0]):
                img = cur_batch_x[i]
                img = np.pad(img, (shift_by_x[i], shift_by_y[i], (0, 0)), mode='constant')
                new_img = img[shift_by_x[i][1]:img.shape[0]-shift_by_x[i][0],
                          shift_by_y[i][1]:img.shape[1]-shift_by_y[i][0]]
                shfd_batch_x_lst.append(new_img)
            shifted_batch_x.append(shfd_batch_x_lst)

            shfd_batch_y_lst = []
            for i in range(cur_batch_y.shape[0]):
                msk = cur_batch_y[i]
                msk = np.pad(msk, (shift_by_x[i], shift_by_y[i], (0, 0)), mode='constant')
                new_msk = msk[shift_by_x[i][1]:msk.shape[0] - shift_by_x[i][0],
                          shift_by_y[i][1]:msk.shape[1] - shift_by_y[i][0]]
                shfd_batch_y_lst.append(new_msk)
            shifted_batch_y.append(shfd_batch_y_lst)

        aug_batch_x.append(shifted_batch_x)
        aug_batch_y.append(shifted_batch_y)

    return np.asarray(aug_batch_x), np.asarray(aug_batch_y), flip_flags, (shifts_by_x, shifts_by_y)


def mean_model_predicts(aug_batch_predicts, aug_batch_y, flip_flags, shifts):
    shifts_by_x, shifts_by_y = shifts
    mean_batch_x = []
    mean_batch_y = []

    for i, flip_flag in enumerate(flip_flags):

        cur_batch_x = np.asarray(
            [np.flip(aug_batch_predicts[j], 0) for j in range(aug_batch_predicts.shape[0])]) if flip_flag else aug_batch_predicts
        cur_batch_y = np.asarray(
            [np.flip(aug_batch_y[j], 0) for j in range(aug_batch_y.shape[0])]) if flip_flag else aug_batch_y


        unshifted_batch_x = []
        unshifted_batch_y = []
        for j in range(4):
            shift_by_x = shifts_by_x[i][j]
            shift_by_y = shifts_by_y[i][j]
            unshfd_batch_x_lst = []
            for k in range(cur_batch_x.shape[i][0]):
                img = cur_batch_x[k]
                img = np.pad(img, (shift_by_x[k], shift_by_y[k], (0, 0)), mode='constant')
                new_img = img[shift_by_x[k][1]:img.shape[0] - shift_by_x[k][0],
                          shift_by_y[k][1]:img.shape[1] - shift_by_y[k][0]]
                shfd_batch_x_lst.append(new_img)
            shifted_batch_x.append(shfd_batch_x_lst)

            shfd_batch_y_lst = []
            for k in range(cur_batch_y.shape[0]):
                msk = cur_batch_y[k]
                msk = np.pad(msk, (shift_by_x[k], shift_by_y[k], (0, 0)), mode='constant')
                new_msk = msk[shift_by_x[k][1]:msk.shape[0] - shift_by_x[k][0],
                          shift_by_y[k][1]:msk.shape[1] - shift_by_y[k][0]]
                shfd_batch_y_lst.append(new_msk)
            shifted_batch_y.append(shfd_batch_y_lst)

        mean_batch_x.append(shifted_batch_x)
        mean_batch_y.append(shifted_batch_y)

    return np.asarray(mean_batch_x), np.asarray(mean_batch_y), flip_flags, (shifts_by_x, shifts_by_y)


if __name__ == '__main__':
    main()
