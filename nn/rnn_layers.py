import numpy as np
from .operations import *
from utils.initializers import *


class Layer(object):
    """
    Layer abstraction
    """

    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False  # Whether there are parameters in this layer that can be trained

    def forward(self, input):
        """Forward pass, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward pass, return gradient to input"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradient of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Gaussian()):
        """Initialization

        # Arguments
            in_features: int, the number of input features
            out_features: int, the numbet of required output features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, input):
        """Forward pass

        # Arguments
            input: numpy array with shape (batch, ..., in_features),
            typically (batch, in_features), or (batch, T, in_features) for sequencical data

        # Returns
            output: numpy array with shape (batch, ..., out_features)
        """
        batch = input.shape[0]
        b_reshaped = self.bias.reshape(
            (1,) * (input.ndim - 1) + self.bias.shape)
        output = np.dot(input, self.weights) + b_reshaped
        return output

    def backward(self, out_grad, input):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            out_grad: numpy array with shape (batch, ..., out_features), gradients to output
            input: numpy array with shape (batch, ..., in_features), same with forward input

        # Returns
            in_grad: numpy array with shape (batch, ..., in_features), gradients to input
        """
        dot_axes = np.arange(input.ndim - 1)
        self.w_grad = np.tensordot(np.nan_to_num(
            input), out_grad, axes=(dot_axes, dot_axes))
        self.b_grad = np.sum(out_grad, axis=tuple(dot_axes))
        in_grad = np.dot(out_grad, self.weights.T)
        return in_grad

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params

        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix + ':' + self.name + '/weights': self.weights,
                prefix + ':' + self.name + '/bias': self.bias
            }
            grads = {
                prefix + ':' + self.name + '/weights': self.w_grad,
                prefix + ':' + self.name + '/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class TemporalPooling(Layer):
    """
    Temporal mean-pooling that ignores NaN
    """

    def __init__(self, name='temporal_pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(TemporalPooling, self).__init__(name=name)

    def forward(self, input):
        """Forward pass

        # Arguments
            input: numpy array with shape (batch, time_steps, units)

        # Returns
            output: numpy array with shape (batch, units)
        """
        mask = ~np.any(np.isnan(input), axis=2)
        output = np.sum(np.nan_to_num(input), axis=1)
        output /= np.sum(mask, axis=1, keepdims=True)
        return output

    def backward(self, out_grad, input):
        """Backward pass

        # Arguments
            out_grad: numpy array with shape (batch, units), gradients to output
            input: numpy array with shape (batch, time_steps, units), same with forward input

        # Returns
            in_grad: numpy array with shape (batch, time_steps, units), gradients to input
        """
        batch, time_steps, units = input.shape
        mask = ~np.any(np.isnan(input), axis=2)
        out_grad = out_grad/np.sum(mask, axis=1, keepdims=True)
        in_grad = np.repeat(out_grad, time_steps, 1).reshape(
            (batch, units, time_steps)).transpose(0, 2, 1)
        in_grad *= ~np.isnan(input)
        return in_grad


class RNNCell(Layer):
    "Only for testing the  backward of onestep rnn"

    def __init__(self, in_features, units, name='rnn_cell', initializer=Gaussian()):
        """
        # Arguments
            in_features: int, the number of input features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(RNNCell, self).__init__(name=name)
        self.trainable = True
        self.cell = RNNCellOp()

        self.kernel = initializer.initialize((in_features, units))
        self.recurrent_kernel = initializer.initialize((units, units))
        self.bias = np.zeros(units)

        self.kernel_grad = np.zeros(self.kernel.shape)
        self.r_kernel_grad = np.zeros(self.recurrent_kernel.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, input):
        """
        # Arguments
            input: [input numpy array with shape (batch, in_features),
                    state numpy array with shape (batch, units)]

        # Returns
            output: numpy array with shape (batch, units)
        """
        output = self.cell.forward(
            input, self.kernel, self.recurrent_kernel, self.bias)
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: numpy array with shape (batch, units), gradients to output
            input: [input numpy array with shape (batch, in_features),
                    state numpy array with shape (batch, units)], same with forward input

        # Returns
            in_grad: [gradients to input numpy array with shape (batch, in_features),
                        gradients to state numpy array with shape (batch, units)]
        """
        in_grad, self.kernel_grad, self.r_kernel_grad, self.b_grad = self.cell.backward(
            out_grad, input, self.kernel, self.recurrent_kernel, self.bias)
        return in_grad

    def update(self, params):
        """Update parameters with new params
        """
        for k, v in params.items():
            if '/kernel' in k:
                self.kernel = v
            elif '/recurrent_kernel' in k:
                self.recurrent_kernel = v
            elif '/bias' in k:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters and gradients

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix + ':' + self.name + '/kernel': self.kernel,
                prefix + ':' + self.name + '/recurrent_kernel': self.recurrent_kernel,
                prefix + ':' + self.name + '/bias': self.bias
            }
            grads = {
                prefix + ':' + self.name + '/kernel': self.kernel_grad,
                prefix + ':' + self.name + '/recurrent_kernel': self.r_kernel_grad,
                prefix + ':' + self.name + '/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class RNN(Layer):
    def __init__(self, in_features, units, h0=None, name='rnn', initializer=Gaussian()):
        """
        # Arguments
            in_features: int, the number of input features
            units: int, the number of hidden units
            h0: default initial state, numpy array with shape (units,)
        """
        super(RNN, self).__init__(name=name)
        self.trainable = True
        self.cell = RNNCellOp()  # it's operation instead of layer

        self.kernel = initializer.initialize((in_features, units))
        self.recurrent_kernel = initializer.initialize((units, units))
        self.bias = np.zeros(units)

        if h0 is None:
            self.h0 = np.zeros_like(self.bias)
        else:
            self.h0 = h0

        self.kernel_grad = np.zeros(self.kernel.shape)
        self.r_kernel_grad = np.zeros(self.recurrent_kernel.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, input):
        output = []
        h = self.h0
        for t in range(input.shape[1]):
            out = self.cell.forward(
                [input[:, t, :], h], self.kernel, self.recurrent_kernel, self.bias)
            output.append(out)
            h = out
        output = np.stack(output, axis=1)
        return output

    def backward(self, out_grad, input):
        output = self.forward(input)
        in_grad = []
        h_grad = np.zeros_like(self.h0)
        self.kernel_grad[:] = 0
        self.r_kernel_grad[:] = 0
        self.b_grad[:] = 0
        for t in range(input.shape[1]-1, -1, -1):
            if t == 0:
                h = np.ones((input.shape[0], 1)).dot(self.h0[None, :])
            else:
                h = output[:, t-1, :]
            grad, kernel_grad, r_kernel_grad, b_grad = self.cell.backward(
                out_grad[:, t, :]+h_grad, [input[:, t, :], h], self.kernel, self.recurrent_kernel, self.bias)
            self.kernel_grad += kernel_grad
            self.r_kernel_grad += r_kernel_grad
            self.b_grad += b_grad
            in_grad.append(grad[0])
            h_grad = grad[1]
        in_grad = np.stack(in_grad, axis=1)
        in_grad = in_grad[:, ::-1, :]
        return in_grad

    def update(self, params):
        """Update parameters with new params
        """
        for k, v in params.items():
            if '/kernel' in k:
                self.kernel = v
            elif '/recurrent_kernel' in k:
                self.recurrent_kernel = v
            elif '/bias' in k:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters and gradients

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/kernel': self.kernel,
                prefix+':'+self.name+'/recurrent_kernel': self.recurrent_kernel,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/kernel': self.kernel_grad,
                prefix+':'+self.name+'/recurrent_kernel': self.r_kernel_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class BidirectionalRNN(Layer):
    """ Bi-directional RNN in Concatenating Mode
    """

    def __init__(self, in_features, units, h0=None, hr=None, name='brnn', initializer=Gaussian()):
        """Initialize two inner RNNs for forward and backward processes, respectively

        # Arguments
            in_features: int, the number of input features
            units: int, the number of hidden units
            h0: default initial state for forward phase, numpy array with shape (units,)
            hr: default initial state for backward phase, numpy array with shape (units,)
        """
        super(BidirectionalRNN, self).__init__(name=name)
        self.trainable = True
        self.forward_rnn = RNN(in_features, units, h0,
                               'forward_rnn', initializer=initializer)
        self.backward_rnn = RNN(in_features, units, hr,
                                'backward_rnn', initializer=initializer)

    def _reverse_temporal_data(self, x, mask):
        """ Reverse a batch of sequence data

        # Arguments
            x: a numpy array of shape (batch, time_steps, units), e.g.
                [[x_0_0, x_0_1, ..., x_0_k1, Unknown],
                ...
                [x_n_0, x_n_1, ..., x_n_k2, Unknown, Unknown]] (x_i_j is a vector of dimension of D)
            mask: a numpy array of shape (batch, time_steps), indicating the valid values, e.g.
                [[1, 1, ..., 1, 0],
                ...
                [1, 1, ..., 1, 0, 0]]

        # Returns
            reversed_x: numpy array with shape (batch, time_steps, units)
        """
        num_nan = np.sum(~mask, axis=1)
        reversed_x = np.array(x[:, ::-1, :])
        for i in range(num_nan.size):
            reversed_x[i] = np.roll(
                reversed_x[i], x.shape[1]-num_nan[i], axis=0)
        return reversed_x

    def forward(self, input):
        """
        Forward pass for concatenating hidden vectors obtained from the RNN 
        processing on normal sentences and the RNN processing on reversed sentences.
        output concatenate the two produced sequences.

        # Arguments
            input: input numpy array with shape (batch, time_steps, in_features), 

        # Returns
            output: numpy array with shape (batch, time_steps, units*2)
        """
        mask = ~np.any(np.isnan(input), axis=2)
        forward_output = self.forward_rnn.forward(input)
        backward_output = self.backward_rnn.forward(
            self._reverse_temporal_data(input, mask))
        output = np.concatenate(
            [forward_output, self._reverse_temporal_data(backward_output, mask)], axis=2)
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: numpy array with shape (batch, time_steps, units*2), gradients to output
            input: numpy array with shape (batch, time_steps, in_features), same with forward input

        # Returns
            in_grad: numpy array with shape (batch, time_steps, in_features), gradients to input
        """
        mask = ~np.any(np.isnan(input), axis=2)
        H = out_grad.shape[2]
        in_grad = self.backward_rnn.backward(self._reverse_temporal_data(out_grad[:, :, H//2:], mask),
                                             self._reverse_temporal_data(input, mask))
        in_grad = self._reverse_temporal_data(in_grad, mask)
        in_grad += self.forward_rnn.backward(out_grad[:, :, :H//2], input)
        return in_grad

    def update(self, params):
        """Update parameters with new params
        """
        for k, v in params.items():
            if '/forward_kernel' in k:
                self.forward_rnn.kernel = v
            elif '/forward_recurrent_kernel' in k:
                self.forward_rnn.recurrent_kernel = v
            elif '/forward_bias' in k:
                self.forward_rnn.bias = v
            elif '/backward_kernel' in k:
                self.backward_rnn.kernel = v
            elif '/backward_recurrent_kernel' in k:
                self.backward_rnn.recurrent_kernel = v
            elif '/backward_bias' in k:
                self.backward_rnn.bias = v

    def get_params(self, prefix):
        """Return parameters and gradients

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/forward_kernel': self.forward_rnn.kernel,
                prefix+':'+self.name+'/forward_recurrent_kernel': self.forward_rnn.recurrent_kernel,
                prefix+':'+self.name+'/forward_bias': self.forward_rnn.bias,
                prefix+':'+self.name+'/backward_kernel': self.backward_rnn.kernel,
                prefix+':'+self.name+'/backward_recurrent_kernel': self.backward_rnn.recurrent_kernel,
                prefix+':'+self.name+'/backward_bias': self.backward_rnn.bias
            }
            grads = {
                prefix+':'+self.name+'/forward_kernel': self.forward_rnn.kernel_grad,
                prefix+':'+self.name+'/forward_recurrent_kernel': self.forward_rnn.r_kernel_grad,
                prefix+':'+self.name+'/forward_bias': self.forward_rnn.b_grad,
                prefix+':'+self.name+'/backward_kernel': self.backward_rnn.kernel_grad,
                prefix+':'+self.name+'/backward_recurrent_kernel': self.backward_rnn.r_kernel_grad,
                prefix+':'+self.name+'/backward_bias': self.backward_rnn.b_grad
            }
            return params, grads
        else:
            return None
