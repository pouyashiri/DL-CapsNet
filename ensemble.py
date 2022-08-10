import numpy as np
import imp
from keras.utils import to_categorical
from load_datasets import *
from deepcaps import DeepCapsNet, DeepCapsNet28
import argparse
from utils import margin_loss, margin_loss_hard, CustomModelCheckpoint

"""
Inputs: n_ensemble, dataset, results_folder
"""

import tensorflow as tf
from keras import optimizers

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

parser = argparse.ArgumentParser(description="Ensemble DeepCFC.")
parser.add_argument('--n_ens', default=7, required=True, type=int)
parser.add_argument('--drp_rate', default='0.0')
parser.add_argument('--dset', default='CIFAR10', required=True)
parser.add_argument('--res_folder', default='results', required=True)

args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = load(args.dset)

true_labels = np.argmax(y_test, 1)
if args.dset in ['CIFAR10', 'SVHN']:
    x_test = resize(x_test, 64)
    x_train = resize(x_train, 64)

tot = None
folder = f'{args.res_folder}/{args.dset}-{args.drp_rate}'
for i in range(1, args.n_ens + 1):
    if args.dset in ['CIFAR10', 'SVHN']:
        # m1 = imp.load_source('module.name', 'deepcaps.py')
        model, eval_model1 = DeepCapsNet(input_shape=x_train.shape[1:], n_class=y_train.shape[1], routings=3,
                                         drp_rate=0.0)
    else:
        model, eval_model1 = DeepCapsNet28(input_shape=x_test.shape[1:], n_class=10, routings=3, drp_rate=0.0)

    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=[margin_loss_hard, 'mse'],
                  loss_weights=[1, 0.4], metrics={'capsnet': "accuracy"})
    eval_model1.load_weights(f'{folder}/{i}/best_weights_2')

    a1, b1 = eval_model1.predict(x_test)
    p = np.sum(np.argmax(a1, 1) == true_labels) / y_test.shape[0]
    print(f'p{i} Test acc:', p)
    if tot is None:
        tot = p
    else:
        tot += p

print(np.shape(tot))
print('Ensemble acc:', np.sum(np.argmax(tot, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
