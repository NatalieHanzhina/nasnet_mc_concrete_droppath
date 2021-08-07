#import gc
import os
import random
import re
import subprocess
import time
import timeit
from subprocess import PIPE

import numpy as np
import nvidia_smi
import tensorflow as tf
from tensorflow.keras.metrics import Mean

from datasets.dsb_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef_combined, hard_dice_coef
from metrics_do import actual_accuracy_and_confidence, brier_score, entropy, compute_filtered_hard_dice, \
    compute_mce_and_correct_ece, compute_FTP_and_FTN, compute_TP_and_TN
from models.model_factory import make_model
from params import args

# import tensorflow_probability as tfp

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)
#test_pred = os.path.join(args.out_root_dir, args.out_masks_folder)
print(args.out_root_dir)

def main():
    retest = False
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    t0 = timeit.default_timer()

    network_args = {'channels': args.channels,
                    'num_workers': args.num_workers,
                    'network': args.network,
                    'resize_size': args.resize_size,
                    'batch_size': args.batch_size,
                    'pretrained_weights': args.pretrained_weights,
                    'dropout_rate': args.dropout_rate,
                    'out_channels': args.out_channels,
                    'test_images_dir': args.test_images_dir,
                    'test_masks_dir': args.test_masks_dir,
                    'out_root_dir': args.out_root_dir,
                    'log_dir': args.log_dir,
                    'models_dir': args.models_dir}

    os.makedirs(args.out_root_dir, exist_ok=True)
    weights = [os.path.join(args.models_dir, m) for m in args.models]
    procs_out = []
    procs_err = []
    patt = re.compile('saved_to_file>>>>>(.+)<<<<<')
    procs = []
    print(f'Using dropout rate {args.dropout_rate}')
    if retest:
        print('Running models')
        nvidia_smi.nvmlInit()
        for w in args.models:
            gpus_free_mem = count_free_gpu_memory()
            while gpus_free_mem[0] < 8.3*2**10:
                time.sleep(100)
                gpus_free_mem = count_free_gpu_memory()
            print("Running model {} from weights {} ".format(args.network, w))
            command = ' '.join(['python predict_test_in_ensemble.py'] + [f'--{k} {v} ' for k, v in network_args.items()])
            command += f' --models {w}'

            #proc = subprocess.Popen(command, shell=True, stdout=PIPE)
            proc = subprocess.Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            procs.append(proc)
            time.sleep(180)


        for proc in procs:
            print('Reading process stderr **************************')
            while True:
                next_err_line = proc.stderr.readline()
                #next_out_line = proc.stdout.readline().decode('utf-8')
                #file_saved_path = re.findall(patt, next_out_line)
                #if len(file_saved_path) > 0:
                #    procs_out.append(file_saved_path[0])
                if proc.poll() is not None:
                    break
                if next_err_line != '':
                    print(next_err_line.decode('utf-8'))
                #if next_out_line != '':
                #    print(next_out_line)

            proc_out, proc_err = proc.communicate()
            print('Process output:-------------')
            print(proc_out.decode('utf-8'))
            print('Process err_out:----------------------------------')
            print(proc_err.decode('utf-8'))
            procs_out.append(re.findall(patt, proc_out.decode('utf-8'))[0])
            procs_err.append(proc_err)
        nvidia_smi.nvmlShutdown()
    else:
        print('Using saved predictions')
        procs_out = ['/mnt/tank/scratch/mkashirin/storage_for_large_models_and_code_from_home/mri_scans_project/src/ensemble_pred/best_nii_mc_sch_dp=0.3_tf_seed=100_nasnet_sch_dp__predicts.npy',
                     '/mnt/tank/scratch/mkashirin/storage_for_large_models_and_code_from_home/mri_scans_project/src/ensemble_pred/best_nii_mc_sch_dp=0.3_tf_seed=200_nasnet_sch_dp__predicts.npy',
                     '/mnt/tank/scratch/mkashirin/storage_for_large_models_and_code_from_home/mri_scans_project/src/ensemble_pred/best_nii_mc_sch_dp=0.3_tf_seed=300_nasnet_sch_dp__predicts.npy',
                     '/mnt/tank/scratch/mkashirin/storage_for_large_models_and_code_from_home/mri_scans_project/src/ensemble_pred/best_nii_mc_sch_dp=0.3_nasnet_sch_dp__predicts.npy']


    print('Loading predictions')
    models_predicts = None
    samples_paths = None
    loss = make_loss(args.loss_function)
    for f_path in procs_out:
        new_pred = np.load(f_path)
        models_predicts = new_pred if models_predicts is None else np.concatenate((models_predicts, new_pred), axis=0)
        print('New_pred shape: ---- ', new_pred.shape)
        new_samp_paths = np.load(f_path.replace('__predicts', '__samples_paths'))
        new_samp_paths = new_samp_paths.reshape((1, new_samp_paths.shape[0]))
        samples_paths = new_samp_paths if samples_paths is None else np.concatenate((samples_paths, new_samp_paths), axis=0)
        print('New_samp shape: ---- ', new_samp_paths.shape)
    print('Predicts shape:', models_predicts.shape)
    print('Samples paths shape:', samples_paths.shape)

    dataset = DSB2018BinaryDataset(args.test_images_dir, args.test_masks_dir, args.channels, seed=args.seed)
    data_generator = dataset.test_ensemble_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)

    data_gen_len = data_generator.__len__()
    ys = []
    for j, ((x, y), samp_path) in enumerate(data_generator):
        ys.append(y[0])
        #print(samples_paths.shape)
        #print(samp_path, samples_paths[:, j])
        assert all(samp_path == samples_paths[:, j])
        if j >= data_gen_len:
            break
    #print(len(ys))

    # counter = -1
    # loop_stop = 20
    # data_gen_len = data_generator.__len__()
    entropy_of_mean = []
    mean_entropy = []
    thrds = (1, 0.8, 0.75, 0.6, 0.5, 0.4, 0.25, 0.2, 0.0)
    metrics = {args.loss_function: [],
               'binary_crossentropy': [],
               'hard_dice_coef_ch1': [],
               'hard_dice_coef': [],
               'hard_dice_coef_combined': [],
               'brier_score': [],
               'expected_calibration_error': [],
               'thresholded hard_dice': [],
               'FTP': [],
               'FTN': [],
               'TP, TN, unc': []
    }
    exclude_metrics = ['tf_brier_score', 'expected_calibration_error', 'maximum_calibration_error',
                       'thresholded hard_dice', 'FTP', 'FTN', 'TP, TN, unc']

    print('Computing metrics')
    prog_bar = tf.keras.utils.Progbar(models_predicts.shape[1])
    for i in range(models_predicts.shape[1]):
        # counter += 1
        # if counter >= loop_stop:
        #     break

        y = ys[i]
        #ensemble_predict = np.asarray([models_predicts[j][i] for j in range(len(models_predicts))])
        ensemble_predict = models_predicts[:, i, ...]
        mean_ens_predict = np.mean(ensemble_predict, axis=0)
        #print('ys shape', np.asarray(ys).shape)
        #print('models_pred shape', models_predicts.shape)
        #print('ensemble_pred shape', ensemble_predict.shape)
        #print('mean_ensemble_pred shape', mean_ens_predict.shape)
        #return

        batch_mean_entropy = tf.reduce_mean(entropy(ensemble_predict), axis=0)
        batch_entropy_of_mean = entropy(mean_ens_predict)
        mutual_info = batch_mean_entropy - batch_entropy_of_mean  # mutual-info describes uncertainty of the model about its predictions

        # mean_predicts = tf.math.reduce_mean(np.asarray(predicts_x), axis=1)
        metrics[args.loss_function].append(loss(y, mean_ens_predict).numpy())
        metrics['binary_crossentropy'].append(binary_crossentropy(y, mean_ens_predict).numpy())
        metrics['hard_dice_coef_ch1'].append(hard_dice_coef_ch1(y, mean_ens_predict).numpy())
        metrics['hard_dice_coef'].append(hard_dice_coef(y, mean_ens_predict).numpy())
        metrics['hard_dice_coef_combined'].append(hard_dice_coef_combined(y, mean_ens_predict).numpy())
        metrics['brier_score'].append(brier_score(y, mean_ens_predict).numpy())

        metrics['expected_calibration_error'].append(actual_accuracy_and_confidence(y.astype(np.int32), mean_ens_predict, mutual_info))
        metrics['thresholded hard_dice'].append(compute_filtered_hard_dice(y, mean_ens_predict, mutual_info))
        FTPs, FTNs = compute_FTP_and_FTN(y, mean_ens_predict, mutual_info)
        metrics['FTP'].append(FTPs)
        metrics['FTN'].append(FTNs)
        metrics['TP, TN, unc'].append([*compute_TP_and_TN(y, mean_ens_predict), mutual_info[..., 0]])

        mean_entropy.append(tf.reduce_mean(entropy(ensemble_predict[..., 0]), axis=0))
        #tf.print('m_e:',tf.shape(mean_entropy[-1]))
        entropy_of_mean.append(entropy(mean_ens_predict[..., 0]))
        #tf.print('e_o_m:',tf.shape(entropy_of_mean[-1]))

        prog_bar.update(i + 1, [(k, round(v[-1], 4)) for k, v in metrics.items() if k not in exclude_metrics])

        #del x, y, ensemble_predict, mean_ens_predict
        #gc.collect()

    loss_value, bce_value, hdc1_value, hdc_value, hdcc_value, brier_score_value = \
        Mean()(metrics[args.loss_function]), \
        Mean()(metrics['binary_crossentropy']), \
        Mean()(metrics['hard_dice_coef_ch1']), \
        Mean()(metrics['hard_dice_coef']), \
        Mean()(metrics['hard_dice_coef_combined']), \
        Mean()(metrics['brier_score'])

    ece_bins = 20
    accs, confds, pred_probs, y_true = zip(*metrics['expected_calibration_error'])
    accs, confds, pred_probs, y_true = np.concatenate(np.asarray(accs), axis=0),\
                                       np.concatenate(np.asarray(confds), axis=0),\
                                       np.concatenate(np.asarray(pred_probs), axis=0),\
                                       np.concatenate(np.asarray(y_true), axis=0)
    mce_value, correct_ece_value = compute_mce_and_correct_ece(accs, confds, ece_bins, pred_probs, y_true)
    #tf.print(tf.convert_to_tensor(eces).shape)
    F_dice = {k: np.mean([metrics['thresholded hard_dice'][j][k] for j in range(len(metrics['thresholded hard_dice']))])
              for k in metrics['thresholded hard_dice'][0].keys()}

    FTPs = {k: np.sum([metrics['FTP'][j][k] for j in range(len(metrics['FTP']))]) for k in metrics['FTP'][0].keys()}
    ratio_of_FTPs = {k: (FTPs[1] - FTPs[k]) / FTPs[1] if FTPs[1] > 0 else 0 for k in FTPs.keys()}
    FTNs = {k: np.sum([metrics['FTN'][j][k] for j in range(len(metrics['FTN']))]) for k in metrics['FTN'][0].keys()}
    ratio_of_FTNs = {k: (FTNs[1] - FTNs[k]) / FTNs[1] if FTNs[1] > 0 else 0 for k in FTNs.keys()}

    tp_tn_unc = np.asarray(metrics['TP, TN, unc'])
    print(tp_tn_unc.shape)
    TPs = {}
    TNs = {}
    for thrd in sorted(thrds):
        TPs[thrd] = np.sum(np.where(tp_tn_unc[:, 2] < thrd, tp_tn_unc[:, 0], 0))
        TNs[thrd] = np.sum(np.where(tp_tn_unc[:, 2] < thrd, tp_tn_unc[:, 1], 0))

    #tf.print(np.asarray(mean_entropy).shape, np.asarray(entropy_of_mean).shape)
    mean_entropy_subtr = np.mean(np.asarray(mean_entropy)-np.asarray(entropy_of_mean))
    #mean_entropy_subtr = tf.reduce_mean(mean_entropy-entropy_of_mean)

    print(f'Dropout rate: {args.dropout_rate}')
    print("\n".join(weights), 'evaluation results:')
    print(f'{args.loss_function}: {loss_value:.4f}, '
          f'binary_crossentropy: {bce_value:.4f}, '
          f'hard_dice_coef_ch1: {hdc1_value:.4f}, '
          f'hard_dice_coef: {hdc_value:.4f}',
          f'hard_dice_coef_combined: {hdcc_value:.4f}')
    print('Monte-Calro estimation')
    print(f'brier_score: {brier_score_value:.4f}, '
          f'\nexp_calibration_error: {correct_ece_value:.4f}',
          f'\nmax_calibration_error: {mce_value:.4f}',
          f'\nDices: ' + '\t'.join([f'{k}: {v:.4f}' for k, v in F_dice.items()]),
          f'\nratios of FTPs: '+'\t'.join([f'{k}: {v:.4f}' for k, v in ratio_of_FTPs.items()]),
          f'\nratios of FTNs: '+'\t'.join([f'{k}: {v:.4f}' for k, v in ratio_of_FTNs.items()]),
          '\n____________________________________'
          f'\nnew formula ratios of TPs: ' + '\t'.join([f'{k}: {v:.4f}' for k, v in TPs.items()]),
          f'\nnew formula ratios of TNs: ' + '\t'.join([f'{k}: {v:.4f}' for k, v in TNs.items()]),
          f'\nmean_entropy_subtr: {mean_entropy_subtr:.4f}')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    exit(0)


def count_free_gpu_memory():
    free_mem_MiB = []
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #nvidia_smi.nvmlInit()  Moved to def main
    for i in range(len(gpus)):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free_MiB = info.free/2**20
        free_mem_MiB.append(free_MiB)

    #nvidia_smi.nvmlShutdown() Moved to def main
    return free_mem_MiB


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
