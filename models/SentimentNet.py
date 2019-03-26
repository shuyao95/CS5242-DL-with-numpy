from nn.rnn_layers import *
from nn.model import Model
from utils.initializers import *

def SentimentNet(word_to_idx):
    """Construct a RNN model for sentiment analysis

    # Arguments:
        word_to_idx: A dictionary giving the vocabulary. It contains V entries,
            and maps each string to a unique integer in the range [0, V).
    # Returns
        model: the constructed model
    """
    vocab_size = len(word_to_idx)

    model = Model()
    model.add(FCLayer(vocab_size, 200, name='embedding', initializer=Gaussian(std=0.01)))
    model.add(BidirectionalRNN(in_features=200, units=50, initializer=Gaussian(std=0.01)))
    model.add(FCLayer(100, 32, name='fclayer1', initializer=Gaussian(std=0.01)))
    model.add(TemporalPooling()) # defined in layers.py
    model.add(FCLayer(32, 2, name='fclayer2', initializer=Gaussian(std=0.01)))
    
    return model