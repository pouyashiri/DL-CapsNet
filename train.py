#  vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from keras import optimizers
from keras.utils import plot_model
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import multi_gpu_model
from load_datasets import *
from utils import margin_loss, margin_loss_hard, CustomModelCheckpoint
from dlcaps import BaseCapsNet, DLCapsNet, DLCapsNet28
import os, time
import imp
import pickle


class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train(model, data, hard_training, args, parse_args, time_callback):
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    # print(f'Appendix = {appendix}')
    # input('contonue?')
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log' + appendix + '.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', batch_size=parse_args.bsize,
                               histogram_freq=int(args.debug), write_grads=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_capsnet_acc', patience=8, restore_best_weights= True)
    # checkpoint1 = CustomModelCheckpoint(model, args.save_dir + '/best_weights_1' + appendix, monitor='val_capsnet_acc',
    #                                     save_best_only=False, save_weights_only=True, verbose=1)

    checkpoint2 = CustomModelCheckpoint(model, args.save_dir + '/best_weights_2' + appendix, monitor='val_capsnet_acc',
                                        save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * 0.5 ** (epoch // 10))

    if (args.numGPU > 1):
        parallel_model = multi_gpu_model(model, gpus=args.numGPU)
    else:
        parallel_model = model

    if (not hard_training):
        parallel_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[margin_loss, 'mse'], loss_weights=[1, 0.4],
                               metrics={'capsnet': "accuracy"})
    else:
        parallel_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[margin_loss_hard, 'mse'],
                               loss_weights=[1, 0.4], metrics={'capsnet': "accuracy"})

    # Begin: Training with data augmentation
    def train_generator(x, y, batch_size, shift_fraction=args.shift_fraction):
        train_datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                           featurewise_std_normalization=False,
                                           samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06,
                                           rotation_range=0.1,
                                           width_shift_range=0.1, height_shift_range=0.1, shear_range=0.0,
                                           zoom_range=0.1, channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
                                           horizontal_flip=True,
                                           vertical_flip=False, rescale=None, preprocessing_function=None,
                                           data_format=None)  # shift up to 2 pixel for MNIST
        train_datagen.fit(x)
        generator = train_datagen.flow(x, y, batch_size=batch_size, shuffle=True)
        while True:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    parallel_model.fit_generator(generator=train_generator(x_train, y_train, parse_args.bsize, args.shift_fraction),
                                 steps_per_epoch=int(y_train.shape[0] / parse_args.bsize), epochs=parse_args.nepoch,
                                 validation_data=[[x_test, y_test], [y_test, x_test]],
                                 callbacks=[lr_decay, log, checkpoint2, time_callback, early_stop],
                                 initial_epoch=int(args.ep_num),
                                 shuffle=True)

    #    parallel_model.save(args.save_dir + '/trained_model_multi_gpu.h5')
    #    model.save(args.save_dir + '/trained_model.h5')

    return parallel_model


def test(eval_model, data, dir):
    (x_train, y_train), (x_test, y_test) = data

    # uncommnt and add the corresponding .py and weight to test other models
    # m1 = imp.load_source('module.name', args.save_dir+"/deepcaps.py")
    # _, eval_model = m1.DeepCapsNet(input_shape=x_test.shape[1:], n_class=10, routings=3)
    # eval_model.load(args.save_dir+"/best_weights_1.h5")
    eval_model.load_weights(args.save_dir + "/best_weights_2x")
    t1 = time.time()
    a1, b1 = eval_model.predict(x_test)
    pickle.dump(a1, open(f'{dir}/predictions.p', 'wb'))
    t2 = time.time() - t1
    # print(f'!@#!@# sizes= {np.shape(a1)}-{np.shape(y_test)}-{y_test.shape[0]}')
    p1 = np.sum(np.argmax(a1, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
    print('Test acc:', p1)
    return p1, t2


import argparse

parser = argparse.ArgumentParser(description="torchDeepCapsNet.")
parser.add_argument('--res_folder', default='results', required=True)
parser.add_argument('--dset', default='CIFAR10', required=True)
parser.add_argument('--bsize', default=64, type=int, required=False)
parser.add_argument('--drp_rate', default=0, type=float, required=False)
parser.add_argument('--nepoch', default=100, type=int, required=False)
parse_args = parser.parse_args()


class args:
    save_dir = ''
    numGPU = 1
    # epochs = 100
    # batch_size = 256
    lr = 0.001
    lr_decay = 0.96
    lam_recon = 0.4
    r = 3
    routings = 3
    shift_fraction = 0.1
    debug = False
    digit = 5
    # save_dir = 'model/CIFAR10/13'
    t = False
    w = None
    ep_num = 0
    # dataset = "CIFAR10"
    # drp_rate = 0.1


# to argParse: dataset, batch_size, drp_rate
# create several folders
# create save_dir out of args

args.save_dir = os.path.join(parse_args.res_folder, f'{parse_args.dset}-{parse_args.drp_rate}')
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

newExp = 0

try:
    with open(os.path.join(f'{args.save_dir}', 'expNum'), 'r') as file:
        newExp = int(file.readline())
except:
    pass

newExp += 1
os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(f'{args.save_dir}', 'expNum'), 'w') as file:
    file.write(str(newExp))

args.save_dir = os.path.join(args.save_dir, str(newExp))
os.makedirs(args.save_dir, exist_ok=True)
# try:
#     os.system("cp deepcaps.py " + args.save_dir + "/deepcaps.py")
# except:
#     print("cp deepcaps.py " + args.save_dir + "/deepcaps.py")


# load data
(x_train, y_train), (x_test, y_test) = load(parse_args.dset)

# x_train,y_train,x_test,y_test = load_tiny_imagenet("tiny_imagenet/tiny-imagenet-200", 200)

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

time_callback = TimeHistory()

if parse_args.dset == "FMNIST":
    model, eval_model = DLCapsNet28(input_shape=x_train.shape[1:], n_class=y_train.shape[1],
                                      routings=args.routings, drp_rate=parse_args.drp_rate)  # for 28*28
else:
    model, eval_model = DLCapsNet(input_shape=x_train.shape[1:], n_class=y_train.shape[1],
                                    routings=args.routings, drp_rate=parse_args.drp_rate)  # for 64*64

# plot_model(model, show_shapes=True,to_file=args.save_dir + '/model.png')


# model.load_weights(args.save_dir + '/best_weights_1.h5')

################  training  #################
appendix = ""
model = train(model=model, data=((x_train, y_train), (x_test, y_test)), hard_training=False, args=args,
              parse_args=parse_args, time_callback=time_callback)
# test(eval_model, ((x_train, y_train), (x_test, y_test)))


# model, eval_model = DeepCapsNet(input_shape=x_train.shape[1:], n_class=y_train.shape[1], routings=args.routings)  # for 64*64


appendix = "x"
model = train(model=model, data=((x_train, y_train), (x_test, y_test)), hard_training=True, args=args,
              parse_args=parse_args, time_callback=time_callback)
#############################################

times = time_callback.times

#################  testing  #################
test_acc, test_time = test(eval_model, ((x_train, y_train), (x_test, y_test)), args.save_dir)
with open(os.path.join(args.save_dir, 'res'), 'w') as resFile:
    resFile.write(f'{test_acc} {model.count_params()} {np.mean(times)} {test_time}')

##############################################
