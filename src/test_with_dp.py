import gc
import os
import random
import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean, categorical_accuracy
from tensorflow.keras.optimizers import RMSprop
import tensorflow_addons as tfa
from collections import Counter
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import np_utils, generic_utils
from tensorflow.keras.utils import to_categorical
from skimage.transform import rescale, resize
from sklearn import metrics as metric

from metrics_do import actual_accuracy_and_confidence, brier_score, entropy, crossentropy, compute_mce_and_correct_ece
from models.model_factory import make_model
from params import args
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy


def pollen_dataset(path="../../cropped_pollen_bayesian", gan=False):
    dataset_dir = path
    nb_classes = 0
    pollen_real = []
    target_real = []
    for d, dirs, files in os.walk(dataset_dir):
        nb_classes += 1
        images = filter(lambda x: x.endswith('.png'), files)
        for f in images:
            path = os.path.join(d, f)
            img = Image.open(path)
            img = np.asarray(img, dtype='uint8')
            img = resize(img, (32, 32),
                         preserve_range=True,
                         anti_aliasing=True)
            # img = change_range(img)
            img /= 127.5
            img -= 1.
            pollen_real.append(img)
            target_real.append(d.split(os.path.sep)[-1])
    pollen_real = np.array(pollen_real)
    target_real = np.array(target_real)
    nb_classes -= 1
    counter = Counter(np.array(target_real))
    print(counter)
    cats = np.unique(target_real)
    di = dict(zip(cats, np.arange(len(cats))))
    pollen_X_train, pollen_X_val, pollen_y_train, pollen_y_val = train_test_split(pollen_real, target_real,
                                                                                  test_size=0.15, random_state=23)
    print(nb_classes)
    target_new = []
    for item in pollen_y_val:
        target_new.append(di[item])
    pollen_y_val = target_new
    pollen_y_val = np.array(pollen_y_val)
    pollen_Y_val = to_categorical(pollen_y_val, nb_classes)
    if gan:
        # dataset_dir = u"./SELF_ATTENTION_GET_65k"
        dataset_dir = u"../../STYLE_GAN_GEN_65k"
        # nb_classes = 0
        pollen_gan = []
        target_gan = []
        for d, dirs, files in os.walk(dataset_dir):
            #     nb_classes += 1
            images = filter(lambda x: x.endswith('.jpeg'), files)
            num = sum(1 for _ in filter(lambda x: x.endswith('.jpeg'), files))
            for i, f in enumerate(images):
                #         if (i < (num - 500) or num < 500):
                path = os.path.join(d, f)
                img = Image.open(path)
                img = np.asarray(img, dtype='uint8')
                img = resize(img, (32, 32),
                             preserve_range=True,
                             anti_aliasing=True)
                # img = change_range(img)
                img /= 127.5
                img -= 1.
                pollen_gan.append(img)
                target_gan.append(d.split(os.path.sep)[-1])
        pollen_gan = np.array(pollen_gan)
        target_gan = np.array(target_gan)
        # nb_classes -= 1
        counter = Counter(np.array(target_gan))
        print(counter)
        pollen = pollen_gan
        target = target_gan
    else:
        ind = np.where(pollen_y_train == 'willow')
        ind = ind[0][400:]
        pollen_X_train = np.delete(pollen_X_train, ind, axis=0)
        pollen_y_train = np.delete(pollen_y_train, ind, axis=0)
        ind = np.where(pollen_y_train == 'birch')
        ind = ind[0][400:]
        pollen_X_train = np.delete(pollen_X_train, ind, axis=0)
        pollen_y_train = np.delete(pollen_y_train, ind, axis=0)
        ind = np.where(pollen_y_train == 'maple')
        ind = ind[0][400:]
        pollen_X_train = np.delete(pollen_X_train, ind, axis=0)
        pollen_y_train = np.delete(pollen_y_train, ind, axis=0)
        pollen = pollen_X_train
        target = pollen_y_train
    target_new = []
    for item in target:
        target_new.append(di[item])
    target = target_new
    target = np.array(target)
    return pollen, target, pollen_X_val, pollen_Y_val, pollen_y_val, nb_classes

def main():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    t0 = timeit.default_timer()
    gpu_id = args.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU.
        try:
            # Currently, memory growth needs to be the same across GPUs.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[int(gpu_id)], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized.
            print('!!!!!!!!!!', e)
    predictions_repetition = args.times_sample_per_test
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

    # dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    # data_generator = dataset.test_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)

    path = args.dir
    gan = False
    pollen, target, pollen_X_val, pollen_Y_val, pollen_y_val, nb_classes = pollen_dataset(path, gan)

    # pollen_X_val, pollen_Y_val, pollen_y_val = pollen_X_val[:3,...], pollen_Y_val[:3,...], pollen_y_val[:3,...]

    initial_learning_rate = args.learning_rate
    optimizer = Adam(lr=initial_learning_rate)

    nll_values, brier_score_values, acc_values, f1_values, mi_values, mce_values, ece_values = [], [], [], [], [], [], []

    print('Predicting test')

    for i, model in enumerate(models):
        print(f'Evaluating {weights[i]} model')
        np.random.seed(23+i)
        random.seed(23+i)
        tf.random.set_seed(23+i)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy",
                                                                                     tfa.metrics.F1Score(
                                                                                         num_classes=nb_classes,
                                                                                         average='macro')])

        metrics = {
                   'categorical_crossentropy': [],
                   "accuracy": [],
                   "f1": [],
                   'brier_score': [],
                   'expected_calibration_error': [],
                   'mutual_info': []
                   }
        exclude_metrics = ['tf_brier_score', 'expected_calibration_error', 'maximum_calibration_error']

        counter = -1
        entropy_of_mean = []
        mean_entropy = []
        prog_bar = tf.keras.utils.Progbar(len(pollen_X_val))
        print("Repetitions", predictions_repetition)
        for ind, (x, y) in enumerate(zip(pollen_X_val, pollen_Y_val)):
            counter += 1


            x_repeated = np.tile(x, (predictions_repetition, 1, 1, 1))
            predicts_x_repeated = model.predict(x_repeated, verbose=0)
            predicts_x = predicts_x_repeated

            mean_predicts = tf.math.reduce_mean(tf.convert_to_tensor(predicts_x, dtype=tf.float32), axis=0)
            # deltas = predicts_x - mean_predicts
            # print("deltas", tf.reduce_min(deltas), tf.reduce_max(deltas))
            batch_mean_entropy = crossentropy(mean_predicts)
            batch_entropy_of_mean = tf.reduce_mean([crossentropy(x) for x in predicts_x], axis=0)
            mutual_info = batch_mean_entropy + batch_entropy_of_mean       # mutual-info describes uncertainty of the model about its predictions

            metrics['mutual_info'].append(mutual_info)
            metrics['categorical_crossentropy'].append(categorical_crossentropy(y, mean_predicts).numpy())
            metrics['accuracy'].append(mean_predicts)
            metrics['f1'].append(np.argmax(mean_predicts))
            metrics['brier_score'].append(brier_score(y, mean_predicts).numpy())
            metrics['expected_calibration_error'].append([categorical_accuracy(y,mean_predicts).numpy(),(1 - mutual_info).numpy(), mean_predicts, y])
            # print(categorical_crossentropy(y,mean_predicts).numpy())
            accs, confds, pred_probs, y_true = zip(*metrics['expected_calibration_error'])
            # print("accs",accs)
            # print("confds",confds)

            #
            # mean_entropy.append(tf.reduce_mean(crossentropy(predicts_x), axis=0))
            # entropy_of_mean.append(crossentropy(mean_predicts))

            prog_bar.update(counter+1)

            del x, y, predicts_x, mean_predicts
            gc.collect()

        nll_value, brier_score_value = \
        Mean()(metrics['categorical_crossentropy']), \
        Mean()(metrics['brier_score'])

        acc_value, f1_value =  \
            Mean()(categorical_accuracy(pollen_Y_val, metrics['accuracy'])).numpy(), \
            metric.f1_score(pollen_y_val, metrics['f1'], average="macro")

        mi_value = Mean()(metrics['mutual_info'])

        ece_bins = 20
        accs, confds, pred_probs, y_true = zip(*metrics['expected_calibration_error'])
        mce_value, ece_value = compute_mce_and_correct_ece(accs, confds, ece_bins, pred_probs, y_true)

        print(f'Performed {predictions_repetition} repetitions per sample')
        print(f'Dropout rate: {args.dropout_rate}')
        print(f'{weights[i]} evaluation results:')

        print(f'categorical_crossentropy: {nll_value:.4f}, ')
        print(acc_value)
        print(f'accuracy: {acc_value:.4f}, ')
        print(f'f1_score: {f1_value:.4f}')

        print('Monte-Calro estimation')
        print(f'brier_score: {brier_score_value:.4f}, ',
              f'\nexp_calibration_error: {ece_value:.4f}',
              f'\nmax_calibration_error: {mce_value:.4f}',
              f'\nmutual_information: {mi_value:.4f}')

        nll_values.append(nll_value)
        brier_score_values.append(brier_score_value)
        acc_values.append(acc_value)
        f1_values.append(f1_value)
        mi_values.append(mi_value)
        mce_values.append(mce_value)
        ece_values.append(ece_value)

    print(f'Average evaluation results:')
    print(
          f'categorical_crossentropy: {np.mean(nll_values):.5f}+-{np.std(nll_values):.5f}, ',
          f'accuracy: {np.mean(acc_values):.5f}+-{np.std(acc_values):.5f}, '
          f'f1_score: {np.mean(f1_values):.5f}+-{np.std(f1_values):.5f}',
    )
    print('Monte-Calro estimation')
    print(f'brier_score: {np.mean(brier_score_values):.5f}+-{np.std(brier_score_values):.5f}, ',
          f'\nexp_calibration_error: {np.mean(ece_values):.5f}+-{np.std(ece_values):.5f}',
          f'\nmax_calibration_error: {np.mean(mce_values):.5f}+-{np.std(mce_values):.5f}',
          f'\nmutual_information: {np.mean(mi_values):.5f}+-{np.std(mi_values):.5f}')
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
