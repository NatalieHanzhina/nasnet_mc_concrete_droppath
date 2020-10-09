import os

import numpy as np
from datasets.dsb_binary import DSB2018BinaryDataset
from losses import binary_crossentropy, make_loss, hard_dice_coef_ch1, hard_dice_coef
from models.model_factory import make_model
from params import args

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(1)
import random

random.seed(1)
import tensorflow as tf

tf.random.set_seed(1)
import timeit
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

    for i, model in enumerate(models):
        print(f'Evaluating {weights[i]} model')
        if args.multi_gpu:
            model = multi_gpu_model(model. len(gpus))

        model.compile(loss=make_loss(args.loss_function),
                  optimizer=optimizer,
                  metrics=[binary_crossentropy, hard_dice_coef_ch1, hard_dice_coef])

        test_loss = model.evaluate(data_generator, verbose=1)
        print(f'{weights[i]} evaluation results:')
        print(list(zip([args.loss_function, 'binary_crossentropy', 'hard_dice_coef_ch1', 'hard_dice_coef'], test_loss)))
        a = 1

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
