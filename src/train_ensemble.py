import datetime
import os
import subprocess
import time

import cv2
import nvidia_smi
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from params import args

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class EnsembleCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index + 1]:
            l.trainable = False


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


def main():
    network_args = {'channels': args.channels,
                    'num_workers': args.num_workers,
                    'network': args.network,
                    'alias': args.alias,
                    'resize_size': args.resize_size,
                    'freeze_till_layer': args.freeze_till_layer,
                    'loss_function': args.loss_function,
                    'optimizer': args.optimizer,
                    'learning_rate': args.learning_rate,
                    'decay': args.decay,
                    'batch_size': args.batch_size,
                    'steps_per_epoch': args.steps_per_epoch,
                    'epochs': args.epochs,
                    'pretrained_weights': args.pretrained_weights,
                    'dropout_rate': args.dropout_rate,
                    'images_dir': args.images_dir,
                    'masks_dir': args.masks_dir,
                    'log_dir': args.log_dir,
                    'models_dir': args.models_dir}  #TODO: REFACTOR!!!

    # def make_ensemble(ensemble_type, network_args, **kwargs):
    if args.ensemble_type == 'cross_validation':
        get_new_bash_instruction = run_cross_valid_ensemble #(network_args, networks_count=args.models_count)
    elif args.ensemble_type == 'random_init':
        get_new_bash_instruction = run_random_init_ensemble #(network_args, networks_count=args.models_count)
    elif args.ensemble_type == 'combined':
        pass
    else:
        raise ValueError(f"Undefined ensemble type: {args.ensemble_type}")

    waiting_for_start_count = args.models_count
    nvidia_smi.nvmlInit()
    running_processes = []
    finished_processes = []
    while waiting_for_start_count > 0:
        gpus_free_mem = count_free_gpu_memory()
        for gpu_id, gpu_free_mem in enumerate(gpus_free_mem):
            if gpu_free_mem > 8*2**10 and waiting_for_start_count > 0:
                command = get_new_bash_instruction(network_args, args.models_count - waiting_for_start_count)
                command = f'CUDA_VISIBLE_DEVICES={gpu_id} ' + command
                print(f'Running: {command}')
                stdout_file_name = os.path.join('logs', args.log_dir, f'{args.ensemble_type}_{args.network}_{args.models_count - waiting_for_start_count}_ensemble_bash_log.txt')
                os.makedirs(os.path.join('logs', args.log_dir), exist_ok=True)
                with open(stdout_file_name, 'w+') as f_out:
                    cur_process = subprocess.Popen(command, shell=True, stdout=f_out, stderr=f_out)

                waiting_for_start_count -= 1
                running_processes.append(cur_process)

        for p_i, running_process in enumerate(running_processes):
            if running_process.poll() is not None:
                finished_processes.append(running_processes.pop(p_i))
        print(f'Awaiting processes: {waiting_for_start_count}\t'
              f'Running processes: {len(running_processes)}\t'
              f'Finised processes: {len(finished_processes)}', end='')
        time.sleep(240)

    nvidia_smi.nvmlShutdown()

    while len(running_processes) > 0:
        print(f'Awaiting processes: {waiting_for_start_count}\t'
              f'Running processes: {len(running_processes)}\t'
              f'Finised processes: {len(finished_processes)}\t'
              f'{datetime.datetime}', end='')
        for p_i, proc in enumerate(running_processes):
            if proc.poll() is not None:
                finished_processes.append(running_processes.pop(p_i))
        time.sleep(60)

    # del model
    # K.clear_session()
    # gc.collect()


def run_cross_valid_ensemble(network_args, network_counter):
    pass


def run_random_init_ensemble(network_args, network_counter):
    command = ' '.join(['python train.py'] + [f'--{k} {v} ' for k, v in network_args.items()])
    return command


if __name__ == '__main__':
    main()
