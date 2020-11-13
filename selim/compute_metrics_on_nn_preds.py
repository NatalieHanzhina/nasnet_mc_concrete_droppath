import gc
import os
import pickle

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
    data_generator = dataset.metrics_compute_genetator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    optimizer = RMSprop(lr=args.learning_rate)
    print('Predicting test')

    for m_i, model in enumerate(models):
        print(f'Working with {weights[m_i]} model')
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
        data_paths = []

        loop_stop = data_generator.__len__()
        counter = -1
        mean_pred_mc = []
        y_s = []
        for x, y, batch_paths in tqdm(data_generator):
            counter += 1
            if counter >= loop_stop:
                break
            y_s.append(y)
            data_paths = data_paths + [p.split('/')[-1] for p in batch_paths]
            x_repeated = np.repeat(x, predictions_repetition, axis=0)
            predicts_x_repeated = model.predict(x_repeated, verbose=0)
            predicts_x = np.asarray(
                [predicts_x_repeated[j * predictions_repetition:(j + 1) * predictions_repetition, ...] for j in
                 range(x.shape[0])])

            mean_predicts = tf.math.reduce_mean(predicts_x, axis=1)
            mean_pred_mc.append(mean_predicts)

            metrics[args.loss_function].append(loss(y, mean_predicts).numpy())
            #metrics['TP'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 1) & (y[:, :, :, 0] > 0)))
            metrics['TP'].append(np.sum((np.where(mean_predicts[:, :, :, 0] > 0.4, 1, 0) == 1) & (y[:, :, :, 0] > 0)))
            #metrics['FP'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 1) & (y[:, :, :, 0] == 0)))
            metrics['FP'].append(np.sum((np.where(mean_predicts[:, :, :, 0] > 0.4, 1, 0) == 1) & (y[:, :, :, 0] == 0)))
            #metrics['TN'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 0) & (y[:, :, :, 0] == 0)))
            metrics['TN'].append(np.sum((np.where(mean_predicts[:, :, :, 0] > 0.4, 1, 0) == 0) & (y[:, :, :, 0] == 0)))
            #metrics['FN'].append(np.sum((np.round(mean_predicts[:, :, :, 0], 0) == 0) & (y[:, :, :, 0] > 0)))
            metrics['FN'].append(np.sum((np.where(mean_predicts[:, :, :, 0] > 0.4, 1, 0) == 0) & (y[:, :, :, 0] > 0)))
            # metrics[args.loss_function].append(loss(y, old_mean_predicts).numpy())
            # metrics['binary_crossentropy'].append(binary_crossentropy(y, old_mean_predicts).numpy())
            # metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, old_mean_predicts).numpy())
            # metrics['hard_dice_coef'].append(hard_dice_coef(y, old_mean_predicts).numpy())
            assert metrics['TP'][-1] + metrics['FP'][-1] + metrics['TN'][-1] + metrics['FN'][-1] == y.shape[0]*y.shape[1]*y.shape[2]

            # print(loss(y, old_mean_predicts).numpy(), loss(y, mean_predicts).numpy())
            # print(binary_crossentropy(y, old_mean_predicts).numpy(), binary_crossentropy(y, mean_predicts).numpy())
            # print(hard_dice_coef_ch1(y, old_mean_predicts).numpy(), hard_dice_coef_ch1(y, mean_predicts).numpy())
            # print(hard_dice_coef(y, old_mean_predicts).numpy(), hard_dice_coef(y, mean_predicts).numpy())

            del x, y, predicts_x, mean_predicts
            gc.collect()

        with open('../predictions/pred_and_y.pikle', 'wb') as f:
            pickle.dump(mean_pred_mc, f)
            pickle.dump(y)

        loss_value = Mean()(metrics[args.loss_function])

        data_paths_set = set(data_paths)
        data_paths_indices = {x: [] for x in data_paths_set}
        for j, p in enumerate(data_paths):
            data_paths_indices[p].append(j)

        ld = 1e-8

        def compute_positive_likelihood_ratio(sensitivity, specificity):
            return sensitivity / (1 - specificity+ld)
        
        def compute_negative_likelihood_ratio(sensitivity, specificity):
            return (1 - sensitivity) / (specificity+ld)

        patient_wise_metrics = {args.loss_function: [],
                   'TP': [],
                   'FP': [],
                   'TN': [],
                   'FN': []}

        for k in data_paths_indices.keys():
            patient_wise_metrics['TP'].append(np.mean([x for i, x in enumerate(metrics['TP']) if i in data_paths_indices[k]]))
            patient_wise_metrics['FP'].append(np.mean([x for i, x in enumerate(metrics['FP']) if i in data_paths_indices[k]]))
            patient_wise_metrics['TN'].append(np.mean([x for i, x in enumerate(metrics['TN']) if i in data_paths_indices[k]]))
            patient_wise_metrics['FN'].append(np.mean([x for i, x in enumerate(metrics['FN']) if i in data_paths_indices[k]]))


        sensitivity1 = np.mean([patient_wise_metrics['TP'][i]/(patient_wise_metrics['TP'][i]+patient_wise_metrics['FN'][i]+ld) for i in range(len(data_paths_indices.keys()))])
        specificity1 = np.mean([patient_wise_metrics['TN'][i]/(patient_wise_metrics['TN'][i]+patient_wise_metrics['FP'][i]+ld) for i in range(len(data_paths_indices.keys()))])
        accuracy1 = np.mean([(patient_wise_metrics['TP'][i] + patient_wise_metrics['TN'][i]) / (
                    patient_wise_metrics['TP'][i] + patient_wise_metrics['FP'][i] + patient_wise_metrics['TN'][i] + patient_wise_metrics['FN'][i]+ld) for i in range(len(data_paths_indices.keys()))])
        positive_likelihood_ratio1 = np.mean(
            [compute_positive_likelihood_ratio(patient_wise_metrics['TP'][i] / (patient_wise_metrics['TP'][i] + patient_wise_metrics['FN'][i]+ld),
                                               patient_wise_metrics['TN'][i] / (patient_wise_metrics['TN'][i] + patient_wise_metrics['FP'][i]+ld)) for i in
            range(len(data_paths_indices.keys()))])

        negative_likelihood_ratio1 = np.mean(
            [compute_negative_likelihood_ratio(patient_wise_metrics['TP'][i] / (patient_wise_metrics['TP'][i] + patient_wise_metrics['FN'][i]+ld),
                                              patient_wise_metrics['TN'][i] / (patient_wise_metrics['TN'] + patient_wise_metrics['FP'][i]+ld)) for i in
            range(len(data_paths_indices.keys()))])

        positive_predictive_value1 = np.mean([patient_wise_metrics['TP'][i] / (patient_wise_metrics['TP'][i] + patient_wise_metrics['FP'][i]+ld) for i in range(len(data_paths_indices.keys()))])
        negative_predictive_value1 = np.mean([patient_wise_metrics['TN'][i] / (patient_wise_metrics['TN'][i] + patient_wise_metrics['FN'][i]+ld) for i in range(len(data_paths_indices.keys()))])
        youdens_index1 = np.mean([patient_wise_metrics['TP'][i]/(patient_wise_metrics['TP'][i]+patient_wise_metrics['FN'][i]+ld) +
                                 patient_wise_metrics['TN'][i]/(patient_wise_metrics['TN'][i]+patient_wise_metrics['FP'][i]+ld) - 1 for i in range(len(data_paths_indices.keys()))])


        sensitivity2 = np.mean(patient_wise_metrics['TP']) / (np.mean(patient_wise_metrics['TP']) + np.mean(patient_wise_metrics['FN']) + ld)
        specificity2 = np.mean(patient_wise_metrics['TN']) / (np.mean(patient_wise_metrics['TN']) + np.mean(patient_wise_metrics['FP']) + ld)
        accuracy2 = (np.mean(patient_wise_metrics['TP']) + np.mean(patient_wise_metrics['TN'])) / (
                np.mean(patient_wise_metrics['TP']) + np.mean(patient_wise_metrics['FP']) + np.mean(patient_wise_metrics['TN']) + np.mean(patient_wise_metrics['FN']) + ld)
        positive_likelihood_ratio2 = sensitivity2 / (1 - specificity2 + ld)

        negative_likelihood_ratio2 = (1 - sensitivity2) / (specificity2 + ld)

        positive_predictive_value2 = np.mean(patient_wise_metrics['TP']) / (np.mean(patient_wise_metrics['TP']) + np.mean(patient_wise_metrics['FP']) + ld)
        negative_predictive_value2 = np.mean(patient_wise_metrics['TN']) / (np.mean(patient_wise_metrics['TN']) + np.mean(patient_wise_metrics['FN']) + ld)
        youdens_index2 = sensitivity2 + specificity2 - 1


        sensitivity3 = np.mean(metrics['TP']) / (np.mean(metrics['TP']) + np.mean(metrics['FN']) + ld)
        specificity3 = np.mean(metrics['TN']) / (np.mean(metrics['TN']) + np.mean(metrics['FP']) + ld)
        accuracy3 = (np.mean(metrics['TP']) + np.mean(metrics['TN'])) / (
                np.mean(metrics['TP']) + np.mean(metrics['FP']) + np.mean(metrics['TN']) + np.mean(metrics['FN']) + ld)
        positive_likelihood_ratio3 = sensitivity3 / (1 - specificity3 + ld)

        negative_likelihood_ratio3 = (1 - sensitivity3) / (specificity3 + ld)

        positive_predictive_value3 = np.mean(metrics['TP']) / (np.mean(metrics['TP']) + np.mean(metrics['FP']) + ld)
        negative_predictive_value3 = np.mean(metrics['TN']) / (np.mean(metrics['TN']) + np.mean(metrics['FN']) + ld)
        youdens_index3 = sensitivity3 + specificity3 - 1


        print(f'Performed {predictions_repetition} repetitions per sample')
        print(f'{weights[m_i]} evaluation results:')
        print(f'{args.loss_function}: {loss_value:.4f}\n'
              f'sensitivity1: {sensitivity1:.4f}\n'
              f'specificity1: {specificity1:.4f}\n'
              f'accuracy1: {accuracy1:.4f}\n'
              f'LR+1: {positive_likelihood_ratio1:.4f}\n'
              f'LR-1: {negative_likelihood_ratio1:.4f}\n'
              f'PPV1: {positive_predictive_value1:.4f}\n'
              f'NPV1: {negative_predictive_value1:.4f}\n'
              f'(Youden’s index1: {youdens_index1:.4f}\n')

        print('\n\n'
              f'sensitivity2: {sensitivity2:.4f}\n'
              f'specificity2: {specificity2:.4f}\n'
              f'accuracy2: {accuracy2:.4f}\n'
              f'LR+2: {positive_likelihood_ratio2:.4f}\n'
              f'LR-2: {negative_likelihood_ratio2:.4f}\n'
              f'PPV2: {positive_predictive_value2:.4f}\n'
              f'NPV2: {negative_predictive_value2:.4f}\n'
              f'(Youden’s index2: {youdens_index2:.4f}\n')

        print('\n\n'
              f'sensitivity3: {sensitivity3:.4f}\n'
              f'specificity3: {specificity3:.4f}\n'
              f'accuracy3: {accuracy3:.4f}\n'
              f'LR+3: {positive_likelihood_ratio3:.4f}\n'
              f'LR-3: {negative_likelihood_ratio3:.4f}\n'
              f'PPV3: {positive_predictive_value3:.4f}\n'
              f'NPV3: {negative_predictive_value3:.4f}\n'
              f'(Youden’s index3: {youdens_index3:.4f}\n')

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
