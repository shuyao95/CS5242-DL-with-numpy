import numpy as np
from utils.tools import img2col
# Attension:
# - Never change the value of input, which will change the result of backward


class operation(object):
    """
    Operation abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operation):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operation):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class fc(operation):
    def __init__(self):
        super(fc, self).__init__()

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        output = np.matmul(input, weights) + bias.reshape(1, -1)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of fc layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)
            bias: numpy array with shape (out_features)

        # Returns
            in_grad: gradient to the forward input of fc layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        b_grad = np.sum(out_grad, axis=0)
        return in_grad, w_grad, b_grad


class conv(operation):
    def __init__(self, conv_params):
        """
        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad = 2 means a 2-pixel border of padded with zeros
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
        """
        super(conv, self).__init__()
        self.conv_params = conv_params

    def forward(self, input, weights, bias):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            output: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - kernel_h + 2*pad) // stride
        out_width = 1 + (in_width - kernel_w + 2*pad) // stride
        output = np.zeros((batch, out_channel, out_height, out_width))

        input_pad = np.pad(input, pad_width=(pad,),
                           mode='constant', constant_values=0)

        # get initial nodes of receptive fields in height and width direction
        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        input_conv = img2col(input_pad, recep_fields_h,
                             recep_fields_w, kernel_h, kernel_w)
        output = np.stack(map(
            lambda x: np.matmul(weights.reshape(out_channel, -1), x) + bias.reshape(-1, 1), input_conv), axis=0)

        output = output.reshape(batch, out_channel, out_height, out_width)
        return output

    def backward(self, out_grad, input, weights, bias):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)
            weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
            bias: numpy array with shape (out_channel)

        # Returns
            in_grad: gradient to the forward input of conv layer, with same shape as input
            w_grad: gradient to weights, with same shape as weights
            b_bias: gradient to bias, with same shape as bias
        """
        kernel_h = self.conv_params['kernel_h']  # height of kernel
        kernel_w = self.conv_params['kernel_w']  # width of kernel
        pad = self.conv_params['pad']
        stride = self.conv_params['stride']
        in_channel = self.conv_params['in_channel']
        out_channel = self.conv_params['out_channel']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - kernel_h + 2*pad) // stride
        out_width = 1 + (in_width - kernel_w + 2*pad) // stride

        input_pad = np.pad(input, pad_width=(pad,),
                           mode='constant', constant_values=0)
        # get initial nodes of receptive fields in height and width direction
        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        input_conv = img2col(input_pad, recep_fields_h,
                             recep_fields_w, kernel_h, kernel_w)
        input_conv_grad = np.stack(map(lambda x: np.matmul(weights.reshape(out_channel, -1).T, x),
                                       out_grad.reshape(batch, out_channel, -1)), axis=0)

        input_pad_grad = np.zeros(
            (batch, in_channel, in_height+2*pad, in_width+2*pad))
        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                input_pad_grad[:, :, i:i+kernel_h, j:j+kernel_w] += \
                    input_conv_grad[:, :, idx].reshape(
                        batch, in_channel, kernel_h, kernel_w)
                idx += 1
        in_grad = input_pad_grad[:, :, pad:pad +
                                 in_height, pad:pad+in_width]
        w_grad = sum(
            list(map(lambda x: np.matmul(x[0], x[1].T), zip(out_grad.reshape(batch, out_channel, -1), input_conv))))
        w_grad = w_grad.reshape(weights.shape)

        b_grad = out_grad.sum(axis=(0, 2, 3))

        return in_grad, w_grad, b_grad


class pool(operation):
    def __init__(self, pool_params):
        """
        # Arguments
            pool_params: dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad = 2 means a 2-pixel border of padding with zeros.
        """
        super(pool, self).__init__()
        self.pool_params = pool_params

    def forward(self, input):
        """
        # Arguments
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            output: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - pool_height + 2*pad) // stride
        out_width = 1 + (in_width - pool_width +
                         2*pad) // stride

        input_pad = np.pad(input, pad_width=(pad,),
                           mode='constant', constant_values=0)

        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        input_pool = img2col(input_pad, recep_fields_h,
                             recep_fields_w, pool_height, pool_width)
        input_pool = input_pool.reshape(
            batch, in_channel, -1, out_height, out_width)
        if pool_type == 'max':
            output = np.max(input_pool, axis=2)
        elif pool_type == 'avg':
            output = np.average(input_pool, axis=2)
        else:
            raise ValueError('Doesn\'t support \'%s\' pooling.' %
                             pool_type)
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
            input: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            in_grad: gradient to the forward input of pool layer, with same shape as input
        """
        pool_type = self.pool_params['pool_type']
        pool_height = self.pool_params['pool_height']
        pool_width = self.pool_params['pool_width']
        stride = self.pool_params['stride']
        pad = self.pool_params['pad']

        batch, in_channel, in_height, in_width = input.shape
        out_height = 1 + (in_height - pool_height + 2*pad) // stride
        out_width = 1 + (in_width - pool_width + 2*pad) // stride

        input_pad = np.pad(input, pad_width=(pad,),
                           mode='constant', constant_values=0)

        recep_fields_h = [stride*i for i in range(out_height)]
        recep_fields_w = [stride*i for i in range(out_width)]

        input_pool = img2col(input_pad, recep_fields_h,
                             recep_fields_w, pool_height, pool_width)
        input_pool = input_pool.reshape(
            batch, in_channel, -1, out_height, out_width)

        if pool_type == 'max':
            input_pool_grad = (input_pool == np.max(input_pool, axis=2, keepdims=True)) * \
                out_grad[:, :, np.newaxis, :, :]

        elif pool_type == 'avg':
            scale = 1 / (pool_height*pool_width)
            input_pool_grad = scale * \
                np.repeat(out_grad[:, :, np.newaxis, :, :],
                          pool_height*pool_width, axis=2)

        input_pool_grad = input_pool_grad.reshape(
            batch, in_channel, -1, out_height*out_width)

        input_pad_grad = np.zeros(input_pad.shape)
        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                input_pad_grad[:, :, i:i+pool_height, j:j+pool_width] += \
                    input_pool_grad[:, :, :, idx].reshape(
                        batch, in_channel, pool_height, pool_width)
                idx += 1
        in_grad = input_pad_grad[:, :, pad:pad+in_height, pad:pad+in_width]
        return in_grad


class dropout(operation):
    def __init__(self, rate, training=True, seed=None):
        """
        # Arguments
            rate: float[0, 1], the probability of setting a neuron to zero
            training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
            seed: int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
        """
        self.rate = rate
        self.seed = seed
        self.training = training
        self.mask = None

    def forward(self, input):
        """
        # Arguments
            input: numpy array with any shape

        # Returns
            output: same shape as input
        """
        if self.training:
            scale = 1/(1-self.rate)
            np.random.seed(self.seed)
            p = np.random.random_sample(input.shape)
            self.mask = (p >= self.rate).astype('int')
            output = input * self.mask * scale
        else:
            output = input
        return output

    def backward(self, out_grad, input):
        """
        # Arguments
            out_grad: gradient to forward output of dropout, same shape as input
            input: numpy array with any shape
            mask: the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

        # Returns
            in_grad: gradient to forward input of dropout, same shape as input
        """
        if self.training:
            scale = 1/(1-self.rate)
            in_grad = scale * self.mask * out_grad
        else:
            in_grad = out_grad
        return in_grad


class softmax_cross_entropy(operation):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad
