
import numpy as np
import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import ortho_weight, uniform_weight, zero_bias

""" Decoder using GRU Recurrent Neural Network. """

def param_init_decoder(options, params, prefix='decoder_gru'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)
    params[_p(prefix,'W')] = W
    
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    params[_p(prefix,'U')] = U
    
    params[_p(prefix,'b')] = zero_bias(2*n_h)

    Wx = uniform_weight(n_x, n_h)
    params[_p(prefix,'Wx')] = Wx
    
    Ux = ortho_weight(n_h)
    params[_p(prefix,'Ux')] = Ux
    
    params[_p(prefix,'bx')] = zero_bias(n_h)    

    return params   
    

def decoder_layer(tparams, state_below, prefix='decoder_gru'):
    
    """ state_below: size of n_steps * n_samples * n_x 
    """

    nsteps = state_below.shape[0]
    n_h = tparams[_p(prefix,'Ux')].shape[1]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')])  + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    
    def _step_slice(x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        u = tensor.nnet.sigmoid(_slice(preact, 1, n_h))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h

        return h
                          
    seqs = [state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [tensor.alloc(numpy_floatX(0.),
                                                             n_samples, n_h)],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                strict=True)
                            
    return rval  
