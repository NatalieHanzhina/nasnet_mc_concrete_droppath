import os

import numpy as np
from datasets.dsb_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef
from models.model_factory import make_model
from tqdm import tqdm
from params import args

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

all_ids = []
all_images = []
all_masks = []

OUT_CHANNELS = args.out_channels


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    predictions_repetition = 2
    t0 = timeit.default_timer()

    weights = [os.path.join(args.models_dir, m) for m in args.models]
    models = []
    for w in weights:
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
                predicts_x.append(model.predict(x, verbose=0))
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
