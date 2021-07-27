import cv2
import gc
import os

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from params import args

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from aug.transforms import aug_mega_hardcore

from datetime import datetime
from tensorflow.keras.losses import binary_crossentropy
#from tensorflow.keras.utils import multi_gpu_model

from datasets.dsb_binary import DSB2018BinaryDataset
from models.model_factory import make_model

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from losses import make_loss, hard_dice_coef, hard_dice_coef_ch1

import tensorflow.keras.backend as K
import tensorflow as tf

if args.tf_seed is not None:
    tf.random.set_seed(args.tf_seed)


class ModelCheckpointMGPU(ModelCheckpoint):
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
    model = make_model(args.network,
                       (None, None, args.channels),
                       pretrained_weights=args.pretrained_weights,
                       do_p=args.dropout_rate,
                       resize_size=(args.resize_size, args.resize_size),
                       total_training_steps=args.epochs*args.steps_per_epoch)
    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        #weights_path = args.weights.format(fold)
        weights_path = os.path.join(args.models_dir, args.weights)
        print('Loading weights from {}'.format(weights_path))
        model.load_weights(weights_path, by_name=True)
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
    train_generator = dataset.train_generator((args.crop_size, args.crop_size), (args.resize_size, args.resize_size), args.preprocessing_function, random_transform, batch_size=args.batch_size)
    val_generator = dataset.val_generator((args.resize_size, args.resize_size), args.preprocessing_function, batch_size=args.batch_size)
    #best_model_file = '{}/best_{}{}_fold{}.h5'.format(args.models_dir, args.alias, args.network,fold)
    best_model_file = '{}/best_{}_{}.h5'.format(args.models_dir, args.alias, args.network)

    best_model = ModelCheckpointMGPU(model, filepath=best_model_file, monitor='val_loss',
                                     verbose=1,
                                     mode='min',
                                     period='epoch',
                                     save_best_only=True,
                                     save_weights_only=True)
    #last_model_file = '{}/last_{}{}_fold{}.h5'.format(args.models_dir, args.alias, args.network,fold)
    last_model_file = '{}/last_{}_{}.h5'.format(args.models_dir, args.alias, args.network)

    last_model = ModelCheckpointMGPU(model, filepath=last_model_file, monitor='val_loss',
                                     verbose=1,
                                     mode='min',
                                     period=args.save_period,
                                     save_best_only=False,
                                     save_weights_only=True)
    # if args.multi_gpu:
    #     model = multi_gpu_model(model, len(gpus))
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
    callbacks = [best_model]

    if args.schedule is not None:
        steps = [(float(step.split(":")[0]), int(step.split(":")[1])) for step in args.schedule.split(",")]
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, steps))
        callbacks.insert(0, lrSchedule)
    tb_log_dir_path = "logs/{}_{}".format(args.log_dir, datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    tb = TensorBoard(tb_log_dir_path)
    print(f"Saving tb logs to {tb_log_dir_path}")
    callbacks.append(tb)
    early_stoping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=17, mode='min')
    callbacks.append(early_stoping)
    steps_per_epoch = len(dataset.train_ids) / args.batch_size + 1
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    validation_data = val_generator
    validation_steps = len(dataset.val_ids) //val_generator.batch_size

    model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            max_queue_size=5,
            verbose=1,
            workers=args.num_workers)

    del model
    K.clear_session()
    gc.collect()


if __name__ == '__main__':
    main()
