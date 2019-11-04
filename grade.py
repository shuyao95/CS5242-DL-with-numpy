import argparse
import sys
import os
import numpy as np
import pandas as pd
import pickle
np.set_printoptions(precision=4)
import subprocess
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import traceback
import warnings
warnings.filterwarnings('ignore')

# import keras for comparison
import keras
from keras import layers

# import tools for checking
from utils.tools import rel_error
from utils.check_grads_cnn import check_grads_layer_error as check_grads_layer_cnn
from utils.check_grads_rnn import check_grads_layer_error as check_grads_layer_rnn

# import from my implementation
from nn.layers import Dropout
from nn.optimizers import Adam, RMSprop

# to store relative errors and scores
errors = {
	'conv_forward': 0,
	# 'conv_backward': 0,
	'pool_forward': 0,
	# 'pool_backward': 0,
	'dropout_forward': 0,
	'dropout_backward': 0,
	'rmsprop': 0,
	'gru_forward': 0,
	'gru_backward': 0,
	'birnn_forward': 0,
}

df_cols = [
	'id',
	'conv_forward',
	# 'conv_backward',
	'pool_forward',
	# 'pool_backward',
	'dropout_forward',
	'dropout_backward',
	'rmsprop',
	'gru_forward',
	'gru_backward',
	'birnn_forward',
	'total'
]

scheme = {
	'conv_forward': 3,
	# 'conv_backward': 0,
	'pool_forward': 3,
	# 'pool_backward': 0,
	'dropout_forward': 1.5,
	'dropout_backward': 1.5,
	'rmsprop': 3,
	'gru_forward': 1,
	'gru_backward': 2,
	'birnn_forward': 3,
}

map_error = {
	'correct': 0,
	'wrong': 1,
}

# check for optimizers
# RMSprop
def rmsprop_check(errors, info):
	update_times = 3
	rmsprop_error = []
	# for no bias correction
	print('-' * 50)
	print("check RMSprop ...")
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
		rmsprop_ck = RMSprop_ck(lr=0.001, bata=0, epsilon=1e-16)
		rmsprop = RMSprop(lr=0.001, bata=0, epsilon=1e-16)
		for i in range(update_times):
			xs = rmsprop_ck.update(xs, xs_grads, i)
			xs_test = rmsprop.update(xs_test, xs_grads_test, i)
			for k, v in xs.items():
				rmsprop_error.append(rel_error(v, xs_test[k]))
	except Exception as e:
		info.update({'rmsprop': e})
		for i in range(update_times*len(xs) - len(rmsprop_error)):
			rmsprop_error.append(1)
		traceback.print_exc()
	errors.update({'rmsprop': rmsprop_error})

# Adam
def adam_check(errors, info):
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
		info.update({'adam': e})
		for i in range(update_times*len(xs) - len(adam_error)):
			adam_error.append(1)
		traceback.print_exc()

	# for bias correction
	print('-' * 50)
	print("check Adam with bias correction...")
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
		adam_ck = Adam_ck(lr=0.001, decay=0, bias_correction=True)
		adam = Adam(lr=0.001, decay=0, bias_correction=True)
		for i in range(update_times):
			xs = adam_ck.update(xs, xs_grads, i)
			xs_test = adam.update(xs_test, xs_grads_test, i)
		for k, v in xs.items():
			adam_error.append(rel_error(v, xs_test[k]))
	except Exception as e:
		info.update({'adam bias': e})
		for i in range(update_times - len(adam_error)):
			adam_error.append(1)

# check for convolutional layer
def conv_check(errors, info):
	print('-' * 50)
	print("check conv forward...")
	# define the input for most of layers
	input_size = (4, 3, 16, 16)
	batch, channel, width, height = input_size
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
		layer = Conv2D_ck(params)
		out = layer.forward(inputs)
		weights = np.transpose(layer.weights, (2, 3, 1, 0))
		keras_conv.layers[0].set_weights([weights, layer.bias])
		keras_out = keras_conv.predict(inputs, batch_size=batch)
		errors.update({'conv_forward': [rel_error(out, keras_out)]})
	except Exception as e:
		info.update({'conv forward': e})
		errors.update({'conv_forward': [1]})
		traceback.print_exc()

	# backward
	# print('-' * 50)
	# print("check conv backward...")
	# out_height = 1 + (height + 2 * params['pad'] - params['kernel_h']) // params['stride']
	# out_width = 1 + (width + 2 * params['pad'] - params['kernel_w']) // params['stride']
	# in_grads = np.random.uniform(size=(batch, params['out_channel'], out_height, out_width))

	# inputs = np.random.uniform(size=input_size)
	# try:
	# 	conv = Convolution_ck(params)
	# 	errors.update({'conv_backward': check_grads_layer(conv, inputs, in_grads)})
	# except Exception as e:
	# 	print('student: %s, error: %s' % (args.root, e))
	# 	errors.update({'conv_backward': [1, 1, 1]})
	# 	traceback.print_exc()

# check for pooling layer
def pool_check(errors, info):
	pool_forward_error = []
	pool_backward_error = []
	# forward for max
	print('-' * 50)
	print("check maxpool forward...")
	# define the input for most of layers
	input_size = (4, 3, 16, 16)
	batch, channel, width, height = input_size
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
		layer = Pool2D_ck(params)
		out = layer.forward(inputs)
		pool_forward_error.append(rel_error(out, keras_out))

	except Exception as e:
		info.update({'pool max forward': e})
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
		layers.AveragePooling2D(
			pool_size=(params['pool_height'], params['pool_width']),
			strides=params['stride'],
			padding='valid',
			data_format='channels_first',
			input_shape=input_size[1:]
		)
	])
	inputs = np.random.uniform(size=input_size)
	try:
		keras_out = keras_pool.predict(inputs, batch_size=batch)
		layer = Pool2D_ck(params)
		out = layer.forward(inputs)
		pool_forward_error.append(rel_error(out, keras_out))

	except Exception as e:
		info.update({'pool avg forward': e})
		pool_forward_error.append(1)
		traceback.print_exc()

	errors.update({'pool_forward': pool_forward_error})

	# backward
	# out_height = 1 + (height - params['pool_height'] + 2 * params['pad']) // params['stride']
	# out_width = 1 + (width - params['pool_width'] + 2 * params['pad']) // params['stride']
	# in_grads = np.random.uniform(size=(batch, channel, out_height, out_width))
	# # for max
	# print('-' * 50)
	# print("check maxpool backward...")
	# params = {
	# 	'pool_type': 'max',
	# 	'pool_height': 3,
	# 	'pool_width': 3,
	# 	'pad': 0,
	# 	'stride': 2,
	# }
	# inputs = np.random.uniform(size=input_size)
	# try:
	# 	pool = Pooling_ck(params)
	# 	pool_backward_error+=check_grads_layer(pool, inputs, in_grads)
	# except Exception as e:
	# 	print('student: %s, error: %s' % (args.root, e))
	# 	pool_backward_error+=[1]
	# 	traceback.print_exc()
	# # for avg
	# print('-' * 50)
	# print("check avgpool backward...")
	# params = {
	# 	'pool_type': 'avg',
	# 	'pool_height': 3,
	# 	'pool_width': 3,
	# 	'pad': 0,
	# 	'stride': 2,
	# }
	# inputs = np.random.uniform(size=input_size)
	# try:
	# 	pool = Pooling_ck(params)
	# 	pool_backward_error+=check_grads_layer(pool, inputs, in_grads)
	# except Exception as e:
	# 	print('student: %s, error: %s' % (args.root, e))
	# 	pool_backward_error+=[1]
	# 	traceback.print_exc()
	# errors.update({'pool_backward': pool_backward_error})

# check for dropout layer
def dropout_check(errors, info):
	print('-' * 50)
	print("check dropout forward...")
	rate = 0.1
	# define the input for most of layers
	input_size = (4, 3, 16, 16)
	batch, channel, width, height = input_size
	inputs = np.random.uniform(size=input_size)
	dropout = Dropout(rate, seed=args.seed)
	myout = dropout.forward(inputs)
	try:
		dropout = Dropout_ck(rate, seed=args.seed)
		out = dropout.forward(inputs)
		errors.update({'dropout_forward': [rel_error(out, myout)]})
	except Exception as e:
		info.update({'dropout forward': e})
		errors.update({'dropout_forward': [1]})

	# backward
	print('-' * 50)
	print("check dropout backward...")
	inputs = np.random.uniform(size=input_size)
	in_grads = np.random.uniform(size=input_size)
	try:
		dropout = Dropout_ck(rate, seed=args.seed)
		# dropout.set_mode(True)
		errors.update({'dropout_backward': check_grads_layer_cnn(dropout, inputs, in_grads)})
	except Exception as e:
		info.update({'dropout backward': e})
		errors.update({'dropout_backward': [1]})
		traceback.print_exc()

# check for gru
def gru_check(errors, info):
	print('-' * 50)
	print("check gru forward...")
	N, D, H = 4, 8, 5
	inputs = np.random.uniform(size=(N, D))
	prev_h = np.random.uniform(size=(N, H))

	# compare with the keras implementation
	keras_inputs = layers.Input(shape=(1, D), name='x')
	keras_prev_h = layers.Input(shape=(H,), name='prev_h')
	keras_rnn = layers.GRU(units=H, use_bias=False, recurrent_activation='sigmoid')(keras_inputs, initial_state=keras_prev_h)
	keras_model = keras.Model(inputs=[keras_inputs, keras_prev_h], 
							outputs=keras_rnn)
	try:
		gru_cell = GRUCell_ck(in_features=D, units=H)
		out = gru_cell.forward([inputs, prev_h])
		keras_model.layers[2].set_weights([gru_cell.kernel, gru_cell.recurrent_kernel])
		keras_out = keras_model.predict_on_batch([inputs[:, None,:], prev_h])
		errors.update({'gru_forward': [rel_error(keras_out, out)]})
	except Exception as e:
		info.update({'gru forward': e})
		errors.update({'gru_forward': [1]})
		traceback.print_exc()

	# backward
	in_grads = np.random.uniform(size=(N, H))
	try:
		gru_cell = GRUCell_ck(in_features=D, units=H)
		errors.update({'gru_backward': check_grads_layer_rnn(gru_cell, [inputs, prev_h], in_grads)})
	except Exception as e:
		info.update({'gru backward': e})
		errors.update({'gru_backward': [1]})
		traceback.print_exc()

# check for birnn
def birnn_check(errors, info):
	print('-' * 50)
	print("check birnn forward...")
	N, T, D, H = 4, 5, 6, 7
	x = np.random.uniform(size=(N, T, D))
	x[0, -1:, :] = np.nan
	x[1, -2:, :] = np.nan
	h0 = np.random.uniform(size=(H,))
	hr = np.random.uniform(size=(H,))
	keras_x = layers.Input(shape=(T, D), name='x')
	keras_h0 = layers.Input(shape=(H,), name='h0')
	keras_hr = layers.Input(shape=(H,), name='hr')
	keras_x_masked = layers.Masking(mask_value=0.)(keras_x)
	keras_rnn = layers.RNN(layers.SimpleRNNCell(H), return_sequences=True)
	keras_brnn = layers.Bidirectional(keras_rnn, merge_mode='concat', name='brnn')(
			keras_x_masked, initial_state=[keras_h0, keras_hr])
	keras_model = keras.Model(inputs=[keras_x, keras_h0, keras_hr],
							outputs=keras_brnn)
	try:
		brnn = BiRNN_ck(in_features=D, units=H, h0=h0, hr=hr)
		out = brnn.forward(x)
		keras_model.get_layer('brnn').set_weights([
			brnn.forward_rnn.kernel,
			brnn.forward_rnn.recurrent_kernel, 
			brnn.forward_rnn.bias,
			brnn.backward_rnn.kernel, 
			brnn.backward_rnn.recurrent_kernel,
			brnn.backward_rnn.bias
		])
		keras_out = keras_model.predict_on_batch([np.nan_to_num(x), np.tile(h0, (N, 1)), np.tile(hr, (N, 1))])
		nan_indices = np.where(np.any(np.isnan(x), axis=2))
		keras_out[nan_indices[0], nan_indices[1],:] = np.nan
		errors.update({'birnn_forward': [rel_error(keras_out, out)]})
	except Exception as e:
		info.update({'birnn forward': e})
		errors.update({'birnn_forward': [1]})
		traceback.print_exc()

# final score
def score(errors, scheme, info):
	rmsprop_check(errors, info)
	conv_check(errors, info)
	pool_check(errors, info)
	dropout_check(errors, info)
	gru_check(errors, info)
	birnn_check(errors, info)

	print('-'*50)
	print('all errors:')
	total_score = 0
	score_list = []
	for k, v in errors.items():
		print('%s:\t'%(k), np.array(v))
		v = (np.array(v) < 1e-5).astype(np.int)
		score = (np.sum(v) / len(v)) * scheme[k]
		total_score += score
		score_list += [score]
	score_list += [total_score]
	print('-'*50)
	print('score:', total_score)
	return score_list

if __name__ == "__main__":
	# define the configurations of CS5242 marking
	parser = argparse.ArgumentParser('Grading for CS5242')
	parser.add_argument('--root', type=str, default='.',
						help='the root directory of students codes')
	parser.add_argument('--seed', type=int, default=1234,
						help='random seed of generating input for inputs')
	parser.add_argument('--error_file', type=str, default='errors.pkl',
						help='the list to store all the errors occurring in the code')
	parser.add_argument('--score_file', type=str, default='scores.csv',
						help='the file to store scores of each student')
	args = parser.parse_args()

	# store all the errors
	if not os.path.exists(args.error_file):
		infos = {}
	else:
		with open(args.error_file, 'rb') as f:
			infos = pickle.load(f)
	# keep the seed the same for all students
	np.random.seed(args.seed)

	# check the import error
	ID = args.root.split('/')[-1]
	infos[ID] = {}
	sys.path.insert(0, os.path.join(args.root, 'codes'))
	print('-' * 50)
	print("check codes structure...")
	try:
		from nn.layers import Conv2D as Conv2D_ck
		from nn.layers import Pool2D as Pool2D_ck
		from nn.layers import Dropout as Dropout_ck
		from nn.layers import GRUCell as GRUCell_ck
		from nn.layers import BiRNN as BiRNN_ck
		# from nn.loss import SoftmaxCrossEntropy
		# from nn.optimizers import Adam as Adam_ck
		from nn.optimizers import RMSprop as RMSprop_ck

		s = score(errors, scheme, infos[ID])
		new_df = pd.DataFrame([[ID]+s], columns=df_cols)
		if os.path.exists(args.score_file):
			df = pd.read_csv(args.score_file).append(new_df, ignore_index=True)
		else:
			df = new_df
		df.to_csv(args.score_file, index=False)
		infos[ID].update({'success': True})
	except Exception as e:
		infos[ID].update({'import': e})

	with open(args.error_file, 'wb') as f:
		pickle.dump(infos, f, protocol=pickle.HIGHEST_PROTOCOL)

