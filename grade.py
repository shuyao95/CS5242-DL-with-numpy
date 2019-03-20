import argparse
import sys
import os
import numpy as np
np.set_printoptions(precision=4)
import subprocess
import copy

import traceback
import warnings
warnings.filterwarnings('ignore')

#  import keras for comparison
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras import layers

# import tools for checking
from utils.tools import rel_error
from utils.check_grads import check_grads_layer_error as check_grads_layer

# import from my implementation
from nn.layers import Dropout
from nn.optimizers import Adam

# to store relative errors and scores
errors = {
	'conv_forward': 0,
	'conv_backward': 0,
	'pool_forward': 0,
	'pool_backward': 0,
	'dropout_forward': 0,
	'dropout_backward': 0,
	'adam': 0
}

scores = {
	'conv_forward': 2,
	'conv_backward': 3,
	'pool_forward': 2,  # 1 for avg and 1 for max
	'pool_backward': 2,  # 1 for avg and 1 for max
	'dropout_forward': 1,
	'dropout_backward': 2,
	'adam': 4  # 2 marks for no bias correction, 2 marks for bias correction
}

map_error = {
	'correct': 0,
	'wrong': 1,
}

# define the configurations of CS5242 marking
parser = argparse.ArgumentParser('Grading for CS5242')
parser.add_argument('--root', type=str, default='.',
                    help='the root directory of students codes')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed of generating input for inputs')
parser.add_argument('--check_file', type=str, default='students_to_check.txt',
                    help='the list of students to check individually')
args = parser.parse_args()

# keep the seed the same for all students
np.random.seed(args.seed)

# check the import error
sys.path.insert(0, os.path.join(args.root, 'codes'))
print('-' * 50)
print("check codes structure...")
try:
	from nn.layers import Convolution as Convolution_ck
	from nn.layers import Pooling as Pooling_ck
	from nn.layers import Dropout as Dropout_ck
	# from nn.loss import SoftmaxCrossEntropy
	from nn.optimizers import Adam as Adam_ck
except Exception as e:
	info = 'student: %s, error: %s' % (args.root, e)
	print(info)
	with open(args.check_file, "a") as file:
		file.write(info+'\n')
	file.close()
	traceback.print_exc()
	sys.exit(0)

# define the input for most of layers
input_size = (4, 3, 16, 16)
batch, channel, width, height = input_size

# check for convolutional layer
# forward
print('-' * 50)
print("check conv forward...")
params = {
	'kernel_h': 3,
	'kernel_w': 3,
	'pad': 0,
	'stride': 2,
	'in_channel': channel,
	'out_channel': 8,
}

keras_conv = keras.Sequential([
	layers.Conv2D(filters=params['out_channel'],
	              kernel_size=(params['kernel_h'], params['kernel_w']),
	              strides=(params['stride'], params['stride']),
	              padding='valid',
	              data_format='channels_first',
	              input_shape=input_size[1:]),
])
inputs = np.random.uniform(size=input_size)
try:
	layer = Convolution_ck(params)
	out = layer.forward(inputs)
	weights = np.transpose(layer.weights, (2, 3, 1, 0))
	keras_conv.layers[0].set_weights([weights, layer.bias])
	keras_out = keras_conv.predict(inputs, batch_size=batch)
	errors.update({'conv_forward': [rel_error(out, keras_out)]})
except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	errors.update({'conv_forward': [1]})
	traceback.print_exc()

# backward
print('-' * 50)
print("check conv backward...")
out_height = 1 + (height + 2 * params['pad'] - params['kernel_h']) // params['stride']
out_width = 1 + (width + 2 * params['pad'] - params['kernel_w']) // params['stride']
in_grads = np.random.uniform(size=(batch, params['out_channel'], out_height, out_width))

inputs = np.random.uniform(size=input_size)
try:
	conv = Convolution_ck(params)
	errors.update({'conv_backward': check_grads_layer(conv, inputs, in_grads)})
except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	errors.update({'conv_backward': [1, 1, 1]})
	traceback.print_exc()

# check for pooling layer
pool_forward_error = []
pool_backward_error = []
# forward for max
print('-' * 50)
print("check maxpool forward...")
params = {
	'pool_type': 'max',
	'pool_height': 3,
	'pool_width': 3,
	'pad': 0,
	'stride': 2,
}

keras_pool = keras.Sequential([
	layers.MaxPooling2D(pool_size=(params['pool_height'], params['pool_width']),
	                    strides=params['stride'],
	                    padding='valid',
	                    data_format='channels_first',
	                    input_shape=input_size[1:])
])
inputs = np.random.uniform(size=input_size)
try:
	keras_out = keras_pool.predict(inputs, batch_size=batch)
	layer = Pooling_ck(params)
	out = layer.forward(inputs)
	pool_forward_error.append(rel_error(out, keras_out))

except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	pool_forward_error.append(1)
	traceback.print_exc()
# forward for avg
print('-' * 50)
print("check avgpool forward...")
params = {
	'pool_type': 'avg',
	'pool_height': 3,
	'pool_width': 3,
	'pad': 0,
	'stride': 2,
}
keras_pool = keras.Sequential([
	layers.AveragePooling2D(pool_size=(params['pool_height'], params['pool_width']),
	                        strides=params['stride'],
	                        padding='valid',
	                        data_format='channels_first',
	                        input_shape=input_size[1:])
])
inputs = np.random.uniform(size=input_size)
try:
	keras_out = keras_pool.predict(inputs, batch_size=batch)
	layer = Pooling_ck(params)
	out = layer.forward(inputs)
	pool_forward_error.append(rel_error(out, keras_out))

except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	pool_forward_error.append(1)
	traceback.print_exc()

errors.update({'pool_forward': pool_forward_error})

# backward
out_height = 1 + (height - params['pool_height'] + 2 * params['pad']) // params['stride']
out_width = 1 + (width - params['pool_width'] + 2 * params['pad']) // params['stride']
in_grads = np.random.uniform(size=(batch, channel, out_height, out_width))
# for max
print('-' * 50)
print("check maxpool backward...")
params = {
	'pool_type': 'max',
	'pool_height': 3,
	'pool_width': 3,
	'pad': 0,
	'stride': 2,
}
inputs = np.random.uniform(size=input_size)
try:
	pool = Pooling_ck(params)
	pool_backward_error+=check_grads_layer(pool, inputs, in_grads)
except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	pool_backward_error+=[1]
	traceback.print_exc()
# for avg
print('-' * 50)
print("check avgpool backward...")
params = {
	'pool_type': 'avg',
	'pool_height': 3,
	'pool_width': 3,
	'pad': 0,
	'stride': 2,
}
inputs = np.random.uniform(size=input_size)
try:
	pool = Pooling_ck(params)
	pool_backward_error+=check_grads_layer(pool, inputs, in_grads)
except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	pool_backward_error+=[1]
	traceback.print_exc()
errors.update({'pool_backward': pool_backward_error})

# check for dropout layer
# forward
print('-' * 50)
print("check dropout forward...")
rate = 0.1
inputs = np.random.uniform(size=input_size)
# dropout = Dropout(rate, seed=args.seed)
# myout = dropout.forward(inputs)
# try:
# 	dropout = Dropout_ck(rate, seed=args.seed)
# 	out = dropout.forward(inputs)
# 	errors.update({'dropout_forward': [rel_error(out, myout)]})
# except Exception as e:
# 	print('student: %s, error: %s' % (args.root, e))
# 	errors.update({'dropout_forward': [1]})

# do not check dropout forward
errors.update({'dropout_forward': [0]})

# backward
print('-' * 50)
print("check dropout backward...")
inputs = np.random.uniform(size=input_size)
in_grads = np.random.uniform(size=input_size)
try:
	dropout = Dropout(rate, seed=args.seed)
	dropout.set_mode(True)
	errors.update({'dropout_backward': check_grads_layer(dropout, inputs, in_grads)})
except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	errors.update({'dropout_backward': [1]})
	traceback.print_exc()

# check for optimizers
# Adam
update_times = 1
adam_error = []
# for no bias correction
print('-' * 50)
print("check Adam without bias correction...")
xs = {
	'weights': np.random.uniform(size=(20, 40)),
	'bias': np.random.uniform(size=(40))
}

xs_grads = {
	'weights': np.random.uniform(size=(20, 40)),
	'bias': np.random.uniform(size=(40))
}
xs_test = copy.deepcopy(xs)
xs_grads_test = copy.deepcopy(xs_grads)

try:
	adam_ck = Adam_ck(lr=0.001, decay=0, epsilon=1e-16)
	adam = Adam(lr=0.001, decay=0, epsilon=1e-16)
	for i in range(update_times):
		xs = adam_ck.update(xs, xs_grads, i)
		xs_test = adam.update(xs_test, xs_grads_test, i)
		for k, v in xs.items():
			adam_error.append(rel_error(v, xs_test[k]))
except Exception as e:
	print('student: %s, error: %s' % (args.root, e))
	for i in range(update_times*len(xs) - len(adam_error)):
		adam_error.append(1)
	traceback.print_exc()

# # for bias correction
# print('-' * 50)
# print("check Adam with bias correction...")
# xs = {
# 	'weights': np.random.uniform(size=(20, 40)),
# 	'bias': np.random.uniform(size=(40))
# }
#
# xs_grads = {
# 	'weights': np.random.uniform(size=(20, 40)),
# 	'bias': np.random.uniform(size=(40))
# }
# xs_test = copy.deepcopy(xs)
# xs_grads_test = copy.deepcopy(xs_grads)
#
# try:
# 	adam_ck = Adam_ck(lr=0.001, decay=0, bias_correction=True)
# 	adam = Adam(lr=0.001, decay=0, bias_correction=True)
# 	for i in range(update_times):
# 		xs = adam_ck.update(xs, xs_grads, i)
# 		xs_test = adam.update(xs_test, xs_grads_test, i)
# 	for k, v in xs.items():
# 		adam_error.append(rel_error(v, xs_test[k]))
# except Exception as e:
# 	print('student: %s, error: %s' % (args.root, e))
# 	for i in range(update_times - len(adam_error)):
# 		adam_error.append(1)

errors.update({'adam': adam_error})

print('-'*50)
print('all errors:')
total_score = 0
for k, v in errors.items():
	print('%s:\t'%(k), np.array(v))
	v = (np.array(v) < 1e-6).astype(np.int)
	score = (np.sum(v) / len(v)) * scores[k]
	total_score += score
print('-'*50)
print('score:', total_score)
