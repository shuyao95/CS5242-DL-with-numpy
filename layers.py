import numpy as np 
from utils.tools import *

class Layer(object):
    """
    
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
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
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        batch = inputs.shape[0]
        outputs = np.matmul(inputs, self.weights) + np.repeat(self.bias.reshape(1, -1), batch, axis=0)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        self.w_grad = np.matmul(inputs.T, in_grads)
        self.b_grad = np.sum(in_grads, axis=0)

        out_grads = np.matmul(in_grads, self.weights.T)
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
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
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class Softmax(Layer):
    def __init__(self, num_class, name='softmax'):
        """Initialization

        # Arguments
            num_class: int, the number of category
        """
        super(Softmax, self).__init__(name=name)
        self.num_class = num_class

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, num_class)

        # Returns
            outputs: numpy array with shape (batch, num_class)
        """
        inputs_shift = inputs - np.max(inputs, axis=1, keepdims=True)
        Z = np.sum(np.exp(inputs_shift), axis=1, keepdims=True)
        log_probs = inputs_shift - np.log(Z)
        outputs = np.exp(log_probs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, num_class), gradients to outputs
            inputs: numpy array with shape (batch, num_class), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, num_class), gradients to inputs 
        """
        inputs_shift = inputs - np.max(inputs, axis=1, keepdims=True)
        Z = np.sum(np.exp(inputs_shift), axis=1, keepdims=True)
        log_probs = inputs_shift - np.log(Z)
        probs = np.exp(log_probs)
        tmp = np.sum(probs*in_grads, axis=-1, keepdims=True)
        out_grads = (in_grads - tmp)*probs
        return out_grads

class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h'] # height of kernel
        self.kernel_w = conv_params['kernel_w'] # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        batch, in_channel, in_height, in_width = inputs.shape
        out_height = 1 + (in_height - self.kernel_h + 2*self.pad) // self.stride
        out_width = 1 + (in_width - self.kernel_w + 2*self.pad) // self.stride
        outputs = np.zeros((batch, self.out_channel, out_height, out_width))

        inputs_pad = np.zeros((batch, self.in_channel, in_height+2*self.pad, in_width+2*self.pad))
        inputs_pad[:,:,self.pad:self.pad+in_height,self.pad:self.pad+in_width] = inputs

        # get initial nodes of receptive fields in height and width direction
        recep_fields_h = [self.stride*i for i in range(out_height)]
        recep_fields_w = [self.stride*i for i in range(out_width)]

        inputs_conv = []
        for i in recep_fields_h:
            for j in recep_fields_w:
                inputs_conv.append(inputs_pad[:,:,i:i+self.kernel_h,j:j+self.kernel_w].reshape(batch, -1))
        inputs_conv = np.transpose(np.array(inputs_conv), (1,0,2))
        for i in range(batch):
            outputs[i,:,:,:] = np.matmul(self.weights.reshape(self.out_channel, -1), inputs_conv[i,:,:].T).reshape((self.out_channel, out_height, out_width))
            bias = np.repeat(self.bias.reshape(self.out_channel, -1), out_height*out_width, axis=-1)
            outputs[i,:,:,:] += bias.reshape(self.out_channel, out_height, out_width)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        batch, in_channel, in_height, in_width = inputs.shape
        out_height = 1 + (in_height - self.kernel_h + 2*self.pad) // self.stride
        out_width = 1 + (in_width - self.kernel_w + 2*self.pad) // self.stride

        inputs_pad = np.zeros((batch, self.in_channel, in_height+2*self.pad, in_width+2*self.pad))
        inputs_pad[:,:,self.pad:self.pad+in_height,self.pad:self.pad+in_width] = inputs
        # get initial nodes of receptive fields in height and width direction
        recep_fields_h = [self.stride*i for i in range(out_height)]
        recep_fields_w = [self.stride*i for i in range(out_width)]

        inputs_conv = []
        for i in recep_fields_h:
            for j in recep_fields_w:
                inputs_conv.append(inputs_pad[:,:,i:i+self.kernel_h,j:j+self.kernel_w].reshape(batch, -1))
        inputs_conv = np.transpose(np.array(inputs_conv), (1,0,2))
        
        inputs_conv_grads = [np.matmul(in_grads[i,:,:,:].reshape((self.out_channel, -1)).T, self.weights.reshape(self.out_channel, -1)) for i in range(batch)]
        inputs_conv_grads = np.array(inputs_conv_grads)

        inputs_pad_grads = np.zeros((batch, self.in_channel, in_height+2*self.pad, in_width+2*self.pad))
        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                inputs_pad_grads[:,:,i:i+self.kernel_h,j:j+self.kernel_w] += inputs_conv_grads[:,idx,:].reshape(batch, self.in_channel, self.kernel_h, self.kernel_w)
                idx += 1 
        out_grads = inputs_pad_grads[:,:,self.pad:self.pad+in_height,self.pad:self.pad+in_width]

        self.w_grad = np.matmul(np.transpose(in_grads, (1, 0, 2, 3)).reshape(self.out_channel, -1), inputs_conv.reshape(-1, self.in_channel*self.kernel_h*self.kernel_w))
        self.w_grad = self.w_grad.reshape(self.out_channel, self.in_channel, self.kernel_h, self.kernel_w)

        self.b_grad = np.sum(in_grads, axis=(0,2,3))

        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
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
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >=0 ) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        batch, in_channel, in_height, in_width = inputs.shape
        out_height = 1 + (in_height - self.pool_height + 2*self.pad) // self.stride
        out_width = 1 + (in_width - self.pool_width + 2*self.pad) // self.stride

        inputs_pad = np.zeros((batch, in_channel, in_height+2*self.pad, in_width+2*self.pad))
        inputs_pad[:,:,self.pad:self.pad+in_height,self.pad:self.pad+in_width] = inputs

        recep_fields_h = [self.stride*i for i in range(out_height)]
        recep_fields_w = [self.stride*i for i in range(out_width)]

        inputs_pool = []
        for i in recep_fields_h:
            for j in recep_fields_w:
                inputs_pool.append(inputs_pad[:,:,i:i+self.pool_height,j:j+self.pool_width].reshape(batch, in_channel, -1))
        inputs_pool = np.transpose(np.array(inputs_pool), (1,2,0,3))
        if self.pool_type == 'max':
            outputs = np.max(inputs_pool, axis=-1)
        elif self.pool_type == 'avg':
            outputs = np.average(inputs_pool, axis=-1)
        else:
            raise ValueError('Doesn\'t support \'%s\' pooling.' %self.pool_type)
        outputs = outputs.reshape(batch, in_channel, out_height, out_width)        
        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        batch, in_channel, in_height, in_width = inputs.shape
        out_height = 1 + (in_height - self.pool_height + 2*self.pad) // self.stride
        out_width = 1 + (in_width - self.pool_width + 2*self.pad) // self.stride

        inputs_pad = np.zeros((batch, in_channel, in_height+2*self.pad, in_width+2*self.pad))
        inputs_pad[:,:,self.pad:self.pad+in_height,self.pad:self.pad+in_width] = inputs

        recep_fields_h = [self.stride*i for i in range(out_height)]
        recep_fields_w = [self.stride*i for i in range(out_width)]

        inputs_pool = []
        for i in recep_fields_h:
            for j in recep_fields_w:
                inputs_pool.append(inputs_pad[:,:,i:i+self.pool_height,j:j+self.pool_width].reshape(batch, in_channel, -1))
        inputs_pool = np.transpose(np.array(inputs_pool), (1,2,0,3))

        inputs_pad_grads = np.zeros(inputs_pad.shape)
        inputs_pool_grads = np.zeros(inputs_pool.shape)

        if self.pool_type == 'max':
            inputs_pool_grads = (inputs_pool==np.max(inputs_pool, axis=-1, keepdims=True))*in_grads.reshape(batch, in_channel, -1, 1)

        elif self.pool_type == 'avg':
            scale = 1 / (self.pool_height*self.pool_width)
            inputs_pool_grads = scale * np.repeat(in_grads.reshape(batch, in_channel, -1, 1), self.pool_height*self.pool_width, axis=-1)

        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                inputs_pad_grads[:,:,i:i+self.pool_height,j:j+self.pool_width] += inputs_pool_grads[:,:,idx,:].reshape(batch, in_channel, self.pool_height, self.pool_width)
                idx += 1
        out_grads = inputs_pad_grads[:,:,self.pad:self.pad+in_height,self.pad:self.pad+in_width]

        return out_grads

class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        if self.training:
            scale = 1/(1-self.ratio)
            np.random.seed(self.seed)
            p = np.random.random_sample(inputs.shape)
            self.mask = (p>=self.ratio).astype('int')
            outputs = inputs * self.mask * scale
        else:
            outputs = inputs
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        if self.training:
            scale = 1/(1-self.ratio)
            inputs_grads = scale * self.mask * in_grads
        else:
            inputs_grads = in_grads
        out_grads = inputs_grads
        return out_grads

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
        
