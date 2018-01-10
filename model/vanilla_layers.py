
#import numpy as np
import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import ortho_weight, uniform_weight, zero_bias

""" Encoder using vanilla Recurrent Neural Network. """

def param_init_decoder(options, params, prefix='decoder_vanilla'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    W = uniform_weight(n_x,n_h)
    params[_p(prefix,'W')] = W
    
    U = ortho_weight(n_h)
    params[_p(prefix,'U')] = U
    
    params[_p(prefix,'b')] = zero_bias(n_h)

    return params
    

def decoder_layer(tparams, state_below, prefix='decoder_vanilla'):
    
    """ state_below: size of n_steps * n_samples * n_x 
    """

    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    
    n_h = tparams[_p(prefix,'U')].shape[0]
        
    def _step_slice(x_, h_, U):
        preact = tensor.dot(h_, U)
        preact += x_
        h = tensor.tanh(preact)

        return h

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                    tparams[_p(prefix, 'b')]

    rval, updates = theano.scan(_step_slice,
                                sequences=[state_below_],
                                outputs_info = [tensor.alloc(numpy_floatX(0.),
                                                             n_samples, n_h)],
                                non_sequences = [tparams[_p(prefix, 'U')]],
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps)
                                
    return rval