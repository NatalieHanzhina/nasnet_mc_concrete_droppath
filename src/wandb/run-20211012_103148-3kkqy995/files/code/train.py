import gc
import os

import cv2
# %matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import pickle
# from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, RMSprop
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import np_utils, generic_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.preprocessing.image
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
from skimage.transform import rescale, resize
import wandb
from wandb.keras import WandbCallback
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix, plot_roc
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from params import args

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# from aug.transforms import aug_mega_hardcore
wandb.login()
print(tf.__version__)
print(keras.__version__)
from datetime import datetime
from tensorflow.keras.losses import binary_crossentropy
#from tensorflow.keras.utils import multi_gpu_model

# from datasets.dsb_binary import DSB2018BinaryDataset
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


# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
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
        print('!!!!!!!!!!',e)

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

def change_range(img):
    OldMin = 0
    OldMax = 255
    NewMin = -1
    NewMax = 1

    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    new_img = int((((img - OldMin) * NewRange) / OldRange) + NewMin)
    return np.asarray(new_img, dtype='uint8')


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
    return pollen, target, pollen_X_val, pollen_Y_val, nb_classes


def main():

    batch_size = args.batch_size
    nb_epoch = args.epochs
    suffix = "stylegan"
    initial_learning_rate = args.learning_rate
    if args.stage == "fine_tuning":
        initial_learning_rate = 1e-3
    elif args.stage == "transfer learning":
        initial_learning_rate = 1e-2
    augmentation = args.augmentation
    path = args.dir
    gan = args.gan
    # gan = False
    print(gan)
    means_sc = []
    means_ac = []
    means_f1 = []
    means_ac_val = []
    means_f1_val = []
    pollen, target, pollen_X_val, pollen_Y_val, nb_classes = pollen_dataset(path, gan)
    cv = 0
    kf = StratifiedKFold(n_splits=5)
    net = args.network
    if gan:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)
        model = make_model(net,
                           (32, 32, args.channels),
                           pretrained_weights=args.pretrained_weights,
                           do_p=args.dropout_rate,
                           resize_size=(args.resize_size, args.resize_size),
                           total_training_steps=nb_epoch * args.steps_per_epoch,
                           classes=nb_classes)
        freeze_model(model, args.freeze_till_layer)
        optimizer = RMSprop(lr=args.learning_rate)
        if args.optimizer:
            if args.optimizer == 'rmsprop':
                optimizer = RMSprop(lr=initial_learning_rate)
            elif args.optimizer == 'adam':
                decay_steps = args.decay
                lr_decayed_fn = (tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate,
                    decay_steps))
                optimizer = Adam(lr=initial_learning_rate)
            elif args.optimizer == 'amsgrad':
                optimizer = Adam(lr=initial_learning_rate, decay=float(args.decay), amsgrad=True)
            elif args.optimizer == 'sgd':
                optimizer = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True, decay=float(args.decay))
        best_model_file = '{}/stylegan_best_{}_{}{}.h5'.format(args.models_dir, args.alias, args.network, (cv + 1))

        best_model = ModelCheckpointMGPU(model, filepath=best_model_file, monitor='val_loss',
                                         verbose=1,
                                         mode='min',
                                         period='epoch',
                                         save_best_only=True,
                                         save_weights_only=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy",
                                                                                     tfa.metrics.F1Score(
                                                                                         num_classes=nb_classes,
                                                                                         average='macro')])
        wandb.init(
            project="pollen-nasnet",
            # Set entity to specify your username or team name
            # ex: entity="carey",
            config={
                "optimizer": optimizer,
                "metric": ["accuracy", "f1"],
                "epoch": nb_epoch,
                "batch_size": batch_size,
                "augmentation": augmentation,
                "intial_lr": initial_learning_rate,
                "weights": best_model_file,
                "lrscheduler": "CosineDecayRestarts"
            },
            tags=[args.network, suffix, args.stage + str(cv + 1)], name='first run')
        X_train = pollen
        y_train = target
        Y_train = to_categorical(y_train, nb_classes)
        model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
                            verbose=1, validation_data=(pollen_X_val, pollen_Y_val),
                            callbacks=[
                                       #                          es,
                                       LearningRateScheduler(lr_decayed_fn),
                                       best_model,
                                       WandbCallback()
                                       ])

        del model
        K.clear_session()
        gc.collect()
    else:
        for train, test in kf.split(pollen, target):
            # for i in range(1):
            print("!-----------------------" + str(cv + 1) + "--------------------------!")

            datagen = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=180,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True)
            model = make_model(net,
                               (32, 32, args.channels),
                               pretrained_weights=args.pretrained_weights,
                               do_p=args.dropout_rate,
                               resize_size=(args.resize_size, args.resize_size),
                               total_training_steps=nb_epoch*args.steps_per_epoch,
                               classes=nb_classes)
            if args.weights is None:
                print('No weights passed, training from scratch')
            else:
                #weights_path = args.weights.format(fold)
                weights_path = os.path.join(args.models_dir, args.weights)
                print('Loading weights from {}'.format(weights_path))
                model.load_weights(weights_path, by_name=True)
            if args.stage == "transfer_learning":
                model.load_weights('{}/{}_best_{}_{}{}.h5'.format(args.models_dir, suffix, args.alias, args.network, 1))
            if args.stage == "fine_tuning":
                model.load_weights("{}/{}_best_{}_{}{}_transfer_learning.h5"
                                   .format(args.models_dir, suffix, args.alias, args.network, (cv + 1)))
            freeze_model(model, args.freeze_till_layer)
            if args.stage == "transfer_learning":
                freeze_model(model, "predictions")
            if args.stage == "fine_tuning":
                freeze_model(model, "input_1")
            optimizer = RMSprop(lr=args.learning_rate)
            if args.optimizer:
                if args.optimizer == 'rmsprop':
                    optimizer = RMSprop(lr=initial_learning_rate)
                elif args.optimizer == 'adam':
                    decay_steps = args.decay
                    lr_decayed_fn = (tf.keras.experimental.CosineDecayRestarts(
                        initial_learning_rate,
                        decay_steps))
                    optimizer = Adam(lr=initial_learning_rate)
                elif args.optimizer == 'amsgrad':
                    optimizer = Adam(lr=initial_learning_rate, decay=float(args.decay), amsgrad=True)
                elif args.optimizer == 'sgd':
                    optimizer = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True, decay=float(args.decay))
            es = EarlyStopping(
                monitor='loss',
                patience=20,
                mode='min'
            )
            best_model_file = '{}/best_{}_{}{}.h5'.format(args.models_dir, args.alias, args.network,(cv + 1))
            if args.stage == "transfer_learning":
                best_model_file = "{}/{}_best_{}_{}{}_transfer_learning.h5".format(args.models_dir, suffix, args.alias, args.network, (cv + 1))
            if args.stage == "fine_tuning":
                best_model_file = "{}/{}_best_{}_{}{}_fine_tuning.h5".format(args.models_dir, suffix, args.alias, args.network, (cv + 1))
            best_model = ModelCheckpointMGPU(model, filepath=best_model_file, monitor='val_loss',
                                             verbose=1,
                                             mode='min',
                                             period='epoch',
                                             save_best_only=True,
                                             save_weights_only=True)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy",
                                                                                         tfa.metrics.F1Score(
                                                                                             num_classes=nb_classes,
                                                                                             average='macro')])
            X_train = pollen[train]
            X_test = pollen[test]
            y_train = target[train]
            y_test = target[test]
            Y_train = to_categorical(y_train, nb_classes)
            Y_test = to_categorical(y_test, nb_classes)

            wandb.init(
                project="pollen-nasnet",
                # Set entity to specify your username or team name
                # ex: entity="carey",
                config={
                    "optimizer": optimizer,
                    "metric": ["accuracy", "f1"],
                    "epoch": nb_epoch,
                    "batch_size": batch_size,
                    "augmentation": augmentation,
                    "intial_lr": initial_learning_rate,
                    "weights": best_model_file,
                    "lrscheduler": "CosineDecayRestarts"
                },
                tags=[args.network, suffix, args.stage+str(cv+1)], name='first run')
            config = wandb.config
            print(X_train.min(), X_train.max())
            datagen.fit(X_train)
            if args.stage == "transfer_learning":
                model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
                                verbose=1, validation_data=(X_test, Y_test),
                                callbacks=[best_model,
                                        # es,
                                           LearningRateScheduler(lr_decayed_fn)
                                           # WandbCallback()
                                           ])
            if args.stage == "fine_tuning":
                model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
                                verbose=1, validation_data=(X_test, Y_test),
                                callbacks=[best_model,
                                                                    es
                                           # LearningRateScheduler(lr_decayed_fn),
                                           # WandbCallback()
                                           ])

            print(best_model_file)
            model.load_weights(best_model_file)
            # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy",
            #                                                                              tfa.metrics.F1Score(
            #                                                                                  num_classes=nb_classes,
            #                                                                                  average='macro')])
            print(Y_test)
            print(model.predict_on_batch(X_test))
            score = model.evaluate(X_test, Y_test, verbose=0)
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
            means_sc.append(score[0])
            means_ac.append(score[1])
            means_f1.append(score[2])
            score = model.evaluate(pollen_X_val, pollen_Y_val, verbose=0)
            print('Val score:', score[0])
            print('Val accuracy:', score[1])
            means_ac_val.append(score[1])
            means_f1_val.append(score[2])
            # wandb.finish()
            cv += 1

            del model
            K.clear_session()
            gc.collect()
        print('Crossvalidate Test F1 score:',np.mean(means_f1))
        print('Crossvalidate Test accuracy:',np.mean(means_ac))
        print('Crossvalidate Val F1 score:',np.mean(means_f1_val))
        print('Crossvalidate Val accuracy:',np.mean(means_ac_val))
        with open('means'+net, 'w+') as thefile:
            for item in range(0,len(means_sc)):
                thefile.write("F1-score=%0.4f Acc=%f\n"  %(means_f1[item], means_ac[item]))
            thefile.write("Mean_F1-score=%0.4f Mean_Acc=%f\n"  %(np.mean(means_f1), np.mean(means_ac)))
            for item in range(0,len(means_ac_val)):
                thefile.write("Val_F1-score=%0.4f Val_Acc=%f\n"  %(means_f1_val[item], means_ac_val[item]))
            thefile.write("Val_Mean_F1-score=%0.4f Val_Mean_Acc=%f\n"  %(np.mean(means_f1_val), np.mean(means_ac_val)))

if __name__ == '__main__':
    main()
