import gc
import time
from datetime import datetime
from multiprocessing import Process

import cv2
import nvidia_smi
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from aug.transforms import aug_mega_hardcore

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from datasets.dsb_binary import DSB2018BinaryDataset
from losses import make_loss, hard_dice_coef, hard_dice_coef_ch1
from models.ensemble_factory import make_ensemble
from params import args


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
    if args.crop_size:
        print('Using crops of shape ({}, {})'.format(args.crop_size, args.crop_size))
    elif args.resize_size:
        print('Using resizes of shape ({}, {})'.format(args.resize_size, args.resize_size))
    else:
        print('Using full size images')
    # if args.multi_gpu:
    #     with tf.device("/cpu:0"):
    #         model = make_model(args.network,
    #                            (None, None, args.channels),
    #                            pretrained_weights=args.pretrained_weights,
    #                            do_p=args.dropout_rate)
    # else:
    network_args = {'network': args.network,
                    'input_shape': (None, None, args.channels),
                    'pretrained_weights': args.pretrained_weights,
                    'do_p': args.dropout_rate,
                    'resize_size': (args.resize_size, args.resize_size),
                    'total_training_steps': args.epochs * args.steps_per_epoch}
    ensemble = make_ensemble(args.ensemble_type, network_args, networks_count=args.models_count)

    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        #weights_path = args.weights.format(fold)
        weights_pattern = args.ensemble_weights_pattern
        print('Loading weights from {} into {}-model ensemble'.format(weights_pattern, args.models_count))
        for m_i, model in enumerate(ensemble):
            model.load_weights(weights_pattern.replace('{}', m_i), by_name=True)
            freeze_model(model, args.freeze_till_layer)
    optimizer = RMSprop(lr=args.learning_rate)
    if args.optimizer:
        if args.optimizer == 'rmsprop':
            optimizer = RMSprop(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'adam':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'amsgrad':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay), amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True, decay=float(args.decay))
    #dataset = DSB2018BinaryDataset(args.images_dir, args.masks_dir, args.labels_dir, fold, args.n_folds, seed=args.seed)
    dataset = DSB2018BinaryDataset(args.images_dir, args.masks_dir, args.channels, seed=args.seed)
    random_transform = aug_mega_hardcore()
    train_generator = dataset.train_generator((args.crop_size, args.crop_size), (args.resize_size, args.resize_size),
                                              args.preprocessing_function, random_transform, batch_size=args.batch_size)
    val_generator = dataset.val_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    #best_model_file = '{}/best_{}{}_fold{}.h5'.format(args.models_dir, args.alias, args.network,fold)
    best_ensemble_file = '{}/best_{}{}_{}{}.h5'.format(args.models_dir, args.alias, args.network, args.ensemble_type, '$#')

    best_ensemble = EnsembleCheckpointMGPU(ensemble, filepath=best_ensemble_file, monitor='val_loss',
                                        verbose=1,
                                        mode='min',
                                        period='epoch',
                                        save_best_only=True,
                                        save_weights_only=True)
    #last_model_file = '{}/last_{}{}_fold{}.h5'.format(args.models_dir, args.alias, args.network,fold)
    last_model_file = '{}/last_{}{}.h5'.format(args.models_dir, args.alias, args.network)

    last_ensemble = EnsembleCheckpointMGPU(ensemble, filepath=last_model_file, monitor='val_loss',
                                        verbose=1,
                                        mode='min',
                                        period=args.save_period,
                                        save_best_only=False,
                                        save_weights_only=True)
    # if args.multi_gpu:
    #     model = multi_gpu_model(model, len(gpus))
    for model in ensemble:
        model.compile(loss=make_loss(args.loss_function),
                      optimizer=optimizer,
                      metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

    def schedule_steps(epoch, steps):
        for step in steps:
            if step[1] > epoch:
                print("Setting learning rate to {}".format(step[0]))
                return step[0]
        print("Setting learning rate to {}".format(steps[-1][0]))
        return steps[-1][0]

    #callbacks = [best_model, last_model]
    callbacks = [best_ensemble]

    if args.schedule is not None:
        steps = [(float(step.split(":")[0]), int(step.split(":")[1])) for step in args.schedule.split(",")]
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, steps))
        callbacks.insert(0, lrSchedule)
    tb_log_dir_path = "logs/{}_{}".format(args.log_dir, datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    tb = TensorBoard(tb_log_dir_path)
    print(f"Saving tb logs to {tb_log_dir_path}")
    callbacks.append(tb)
    steps_per_epoch = len(dataset.train_ids) / args.batch_size + 1
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    validation_data = val_generator
    validation_steps = len(dataset.val_ids) // val_generator.batch_size

    fit_lambda = lambda model: model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            max_queue_size=5,
            verbose=1,
            workers=args.num_workers)

    ensemble_processes = []
    for model in ensemble:
        ensemble_processes.append(Process(target=fit_lambda, args=(model,)))

    waiting_for_start = ensemble_processes.copy()
    nvidia_smi.nvmlInit()
    running_processes = []
    finished_processes = []
    while len(waiting_for_start) > 0:
        gpus_free_mem = count_free_gpu_memory()
        for gpu_id, gpu_free_mem in enumerate(gpus_free_mem):
            if gpu_free_mem > 8*2**10:
                process_to_start = waiting_for_start.pop()
                with tf.device(f'/device:gpu:{gpu_id}'):
                    process_to_start.start()
                running_processes.append(process_to_start)
        for p_i, running_process in enumerate(running_processes):
            if not running_process.is_alive():
                finished_processes.append(running_processes.pop(p_i))
        print(f'Awaiting processes: {len(waiting_for_start)}\n'
              f'Running processes: {len(running_processes)}\n'
              f'Finised processes: {len(finished_processes)}')
        time.sleep(60)

    nvidia_smi.nvmlShutdown()

    del model
    K.clear_session()
    gc.collect()


if __name__ == '__main__':
    main()


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

a = tf.random.uniform((1024, 1024, 1024))


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000*9)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

tf.random.uniform((1024, 1024, 1024))