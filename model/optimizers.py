import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor.shared_randomstreams
from utils import numpy_floatX
from collections import OrderedDict
import logging

def SGD(tparams, cost, inps, lr,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):       
        updated_p = p - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def Momentum(tparams, cost, inps, lr, momentum=0.9,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup) 
    
    updates = []

    for p, g in zip(tparams.values(), gshared): 
        m = theano.shared(p.get_value() * 0.)
        m_new = momentum * m - lr * g
        updates.append((m, m_new))        
        
        updated_p = p + m_new
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def NAG(tparams, cost, inps, lr, momentum=0.9,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup) 
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        m_new = momentum * m - lr * g
        updates.append((m, m_new))        
        
        updated_p = p + momentum * m_new - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 
          
def Adagrad(tparams, cost, inps, lr, epsilon=1e-6,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)    
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_t = acc + g ** 2
        updates.append((acc, acc_t))
        p_t = p - (lr / tensor.sqrt(acc_t + epsilon)) * g
        updates.append((p, p_t))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def Adadelta(tparams, cost, inps, lr, rho=0.95, epsilon=1e-6,clip_norm=5):
    """ default: lr=0.5 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_delta = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc,acc_new)) 
        
        update = g * tensor.sqrt(acc_delta + epsilon) / tensor.sqrt(acc_new + epsilon)
        updated_p = p - lr * update
        updates.append((p, updated_p))
        
        acc_delta_new = rho * acc_delta + (1 - rho) * update ** 2
        updates.append((acc_delta,acc_delta_new))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 


def RMSprop_v1(tparams, cost, inps, lr, rho=0.9, epsilon=1e-6,clip_norm=5):
    """ default: lr=0.001 
        This is the implementation of the RMSprop algorithm used in
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        updated_p = p - lr * (g / tensor.sqrt(acc_new + epsilon))
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update
        
def RMSprop_v2(tparams, cost, inps, lr, rho=0.95, momentum=0.9, epsilon=1e-4, clip_norm=5):
    """ default: lr=0.0001 
        This is the implementation of the RMSprop algorithm used in
        http://arxiv.org/pdf/1308.0850v5.pdf
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)    
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc2 = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1.-rho) * g
        acc2_new = rho * acc + (1.-rho) * (g ** 2)
        updates.append((acc, acc_new))
        updates.append((acc2, acc2_new))
        
        updir = theano.shared(p.get_value() * 0.)
        updir_new = momentum * updir - lr * g / tensor.sqrt(acc2_new -acc_new ** 2 + epsilon)
        updates.append((updir, updir_new))
        
        updated_p = p + updir_new
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 
      
def Adam(tparams, cost, inps, lr, b1=0.1, b2=0.001, e=1e-8, clip_norm=5):
    """ default: lr=0.0002 
        This is the implementation of the Adam algorithm
        Reference: http://arxiv.org/pdf/1412.6980v8.pdf
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
    
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    i = theano.shared(numpy_floatX(0.))    
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 


#a=1
def SGMGNHT_1(tparams, cost, inps, ntrain, lr, iterations, rho=0.9, epsilon=1e-6, resamp = 50, clip_norm=1):
    """ Additional parameters """
    mom_tparams = OrderedDict()
    xi_tparams = OrderedDict()
    for k, p0 in tparams.iteritems():
        mom_tparams[k] = theano.shared(p0.get_value() * 0. + 1e-10, name='%s_mom'%k) 
        xi_tparams[k] = theano.shared(p0.get_value() * 0. + 1e-10, name='%s_xi'%k) 
    
    #a = theano.shared(numpy_floatX(1.))
    m = theano.shared(numpy_floatX(1.))
    c = theano.shared(numpy_floatX(5.))
    sigma_p = theano.shared(numpy_floatX(0.01))
    sigma_mom = theano.shared(numpy_floatX(10.))
    sigma_xi = theano.shared(numpy_floatX(0.01))
    gamma = theano.shared(numpy_floatX(0.001))
    logger = logging.getLogger('eval_ptb_sgmgnht')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_ptb_sgmgnht.log')
    logger.info('a = 1, m {} c {} s_p{} s_mom{} s_xi{} g_xi{}'.format( m.get_value(), c.get_value(), sigma_p.get_value(), sigma_mom.get_value(), sigma_xi.get_value(), gamma.get_value()))
    
    p = tensor.vector('p', dtype='float32')
    
    """ default: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    
    # clip norm
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
    # clip value
#    grads = tensor.switch(tensor.ge(grads,clip_norm), clip_norm, grads)
        
    gshared = [theano.shared(p0.get_value() * 0., name='%s_grad'%k) 
                for k, p0 in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
 
    for p, mom, xi, g in zip(tparams.values(),mom_tparams.values(),xi_tparams.values(), gshared):
        
        g_f = mom/m
        #K_f = -g_f + 2/c*(c*g_f + tensor.log(1+tensor.exp(-c*g_f)))        
        K_f = g_f + 2/c*(tensor.log(1+tensor.exp(-c*g_f)))      
        #psi_f_1 = (1- tensor.exp(-c*g_f) )/( 1 + tensor.exp(-c*g_f) )
        
        #psi_f_1 = -1 + 2/( 1 + tensor.exp(-c*g_f) )
	psi_f_1 = -1 + 2*tensor.nnet.sigmoid(c*g_f)
        f1_f_1 = 1/m*psi_f_1
        #psi_grad_f_1 = 2*c*tensor.exp(- c*g_f)/(1 + tensor.exp(-c*g_f))**2
	psi_grad_f_1 = 2*c*tensor.nnet.sigmoid(c*g_f)*(1-tensor.nnet.sigmoid(c*g_f))
        f3_f_1 =  1/m**2*(psi_f_1**2-psi_grad_f_1)
        
#        psi_f = 1- 2/(tensor.exp(c*g_f) + 1)
#        f1_f = 1/m*psi_f
#        psi_grad_f = 2*c*tensor.exp(c*g_f)/(tensor.exp(c*g_f) + 1)**2
#        f3_f =  1/m**2*(psi_f**2-psi_grad_f)
 
        #temp_f1 = tensor.switch(tensor.ge(g_f,0), f1_f_1, f1_f)
        #temp_f3 = tensor.switch(tensor.ge(g_f,0), f3_f_1, f3_f)       
	temp_f1 = f1_f_1
	temp_f3 = f3_f_1

        noise_p = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        noise_mom = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        noise_xi = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        # generata gamma(a,2): N(0,1)^2 = gamma(1/2,2)
        noise_temp = tensor.zeros(p.get_value().shape)
        for aa in xrange(2):
            this_noise = trng.normal(p.get_value().shape, avg = 0.0, std = 1., dtype='float32')
            noise_temp = tensor.inc_subtensor(noise_temp[:], this_noise**2)
        randmg = (noise_temp*m/2)*tensor.sgn(trng.normal(p.get_value().shape, avg = 0.0, std = 1., dtype='float32'))    

        
        updated_p = p +  temp_f1 * lr - g * lr * ntrain * sigma_p + tensor.sqrt(2*sigma_p*lr) * noise_p
        updated_mom = (mom - temp_f1* xi *lr  - g * lr * ntrain + tensor.sqrt(2*sigma_mom*lr) * noise_mom)* (1-tensor.eq(tensor.mod(iterations,resamp),0)) + randmg * tensor.eq(tensor.mod(iterations,resamp),0)
        #updated_mom = mom - temp_f1* xi *lr  - g * lr * ntrain + tensor.sqrt(2*sigma_p*lr) * noise_p
        temp_xi = trng.normal(p.get_value().shape, avg = sigma_mom, std = tensor.sqrt(sigma_xi/2) , dtype='float32')
        updated_xi = (xi + temp_f3* gamma * lr - (xi - sigma_mom)*sigma_xi/(gamma+1e-10)*lr + tensor.sqrt(2*sigma_xi*lr) * noise_xi) * (1-tensor.eq(tensor.mod(iterations,resamp),resamp/2)) + temp_xi * tensor.eq(tensor.mod(iterations,resamp),resamp/2)
        

        updates.append((p, updated_p))
        updates.append((mom, updated_mom))
        updates.append((xi, updated_xi))
    
    f_update = theano.function([lr,ntrain,iterations], [p,mom,xi], updates=updates)
    #f_params = theano.function([], [a, m, c, mom.shape])
    return f_grad_shared, f_update
    
#a = 2
def SGMGNHT_2(tparams, cost, inps, ntrain, lr, iterations, rho=0.9, epsilon=1e-6, resamp = 50, clip_norm=1):
    """ Additional parameters """
    mom_tparams = OrderedDict()
    xi_tparams = OrderedDict()
    #rng = np.random.RandomState(3435)
    #+ rng.normal(0,1,p0.shape())
    for k, p0 in tparams.iteritems():
        mom_tparams[k] = theano.shared(p0.get_value() * 0. +1e-1, name='%s_mom'%k) 
        xi_tparams[k] = theano.shared(p0.get_value() * 0. + 10.0, name='%s_xi'%k) 
    
    #a = theano.shared(numpy_floatX(2.))
    # m = theano.shared(numpy_floatX(1.))
    # c = theano.shared(numpy_floatX(1.))
    # sigma_p = theano.shared(numpy_floatX(10.))
    # sigma_xi = theano.shared(numpy_floatX(0.01))
    # sigma_theta = theano.shared(numpy_floatX(0.1))
    # gamma = theano.shared(numpy_floatX(1.))
    
    m = theano.shared(numpy_floatX(1.))
    c = theano.shared(numpy_floatX(3.))
    sigma_p = theano.shared(numpy_floatX(0.01))
    sigma_mom = theano.shared(numpy_floatX(10.))
    sigma_xi = theano.shared(numpy_floatX(0.01))
    gamma = theano.shared(numpy_floatX(1.0))
    
    logger = logging.getLogger('eval_ptb_sgmgnht')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_ptb_sgmgnht.log')
    logger.info('a = 1, m {} c {} s_p{} s_mom{} s_xi{} g_xi{}'.format( m.get_value(), c.get_value(), sigma_p.get_value(), sigma_mom.get_value(), sigma_xi.get_value(), gamma.get_value()))
    
    p = tensor.vector('p', dtype='float32')
    
    """ default: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    
    # clip norm
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p0.get_value() * 0., name='%s_grad'%k) 
                for k, p0 in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
 
    for p, mom, xi, g in zip(tparams.values(),mom_tparams.values(),xi_tparams.values(), gshared):
        
        g_f = (tensor.sqrt(tensor.abs_(mom+1e-100)))/m
        K_f = g_f + 4/c/(1 + tensor.exp(c*g_f))
        
        psi_f_1 = -1 + 2/( 1 + tensor.exp(-c*g_f))
        f1_f_1 = 1/2.0/m**2 *psi_f_1**2 /g_f*tensor.sgn(mom)
        #f1_f_1 = 1/2.0/m*psi_f_1**2* tensor.abs_(mom+1e-100)**(-1/2) *tensor.sgn(mom)
        psi_grad_f_1 = 2*c*tensor.exp(- c*g_f)/(1 + tensor.exp(-c*g_f))**2
        f3_f_1 =  f1_f_1**2 - 1/2.0/m**2 * psi_f_1 * psi_grad_f_1 / tensor.abs_(mom) + 1/4.0/m * psi_f_1**2 * (tensor.abs_(mom+1e-100)**(-1.5))
        
        
#        psi_f = (tensor.exp(c*g_f) - 1)/(tensor.exp(c*g_f) + 1)
#        f1_f = 1/2/m*psi_f**2 * (tensor.abs_(mom+1e-100)**(-1/2))*tensor.sgn(mom)
#        psi_grad_f = 2*c*tensor.exp(c*g_f)/(tensor.exp(c*g_f) + 1)**2
#        f3_f =  f1_f**2 - c/2/m**2 * psi_f * psi_grad_f / tensor.abs_(mom) + 1/4/m * psi_f**2 * (tensor.abs_(mom+1e-100)**(-3/2))
 
#        temp_f1 = tensor.switch(tensor.ge(g_f,0), f1_f_1, f1_f)
#        temp_f3 = tensor.switch(tensor.ge(g_f,0), f3_f_1, f3_f)       

        temp_f1 = f1_f_1
        temp_f3 = f3_f_1
        
        noise_p = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        noise_mom = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        noise_xi = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')

        # generata gamma(a,2): N(0,1)^2 = gamma(1/2,2)
        noise_temp = tensor.zeros(p.get_value().shape)
        for aa in xrange(4):
            this_noise = trng.normal(p.get_value().shape, avg = 0.0, std = 1., dtype='float32')
            noise_temp = tensor.inc_subtensor(noise_temp[:], this_noise**2)
        randmg = (noise_temp*m/2)**2*tensor.sgn(trng.normal(p.get_value().shape, avg = 0.0, std = 1., dtype='float32'))    

        updated_p = p +  temp_f1 * lr - g * lr * ntrain * sigma_p + tensor.sqrt(2*sigma_p*lr) * noise_p
        updated_mom = (mom - temp_f1* xi *lr  - g * lr * ntrain + tensor.sqrt(2*sigma_mom*lr) * noise_mom)* (1-tensor.eq(tensor.mod(iterations,resamp),0)) + randmg * tensor.eq(tensor.mod(iterations,resamp),0)
        #updated_mom = mom - temp_f1* xi *lr  - g * lr * ntrain + tensor.sqrt(2*sigma_p*lr) * noise_p
        temp_xi = trng.normal(p.get_value().shape, avg = sigma_mom, std = tensor.sqrt(sigma_xi/2) , dtype='float32')
        updated_xi = (xi + temp_f3* gamma * lr - (xi - sigma_mom)*sigma_xi/(gamma+1e-10)*lr + tensor.sqrt(2*sigma_xi*lr) * noise_xi) * (1-tensor.eq(tensor.mod(iterations,resamp),resamp/2)) + temp_xi * tensor.eq(tensor.mod(iterations,resamp),resamp/2)

        updates.append((p, updated_p))
        updates.append((mom, updated_mom))
        updates.append((xi, updated_xi))
    
    f_update = theano.function([lr,ntrain,iterations], [p,mom,xi], updates=updates)
    #f_params = theano.function([], [a, m, c, mom.shape])
    return f_grad_shared, f_update




def SGMGHMC_p(tparams, cost, inps, ntrain, lr, rho=0.9, epsilon=1e-6, clip_norm=0.1):
    """ Additional parameters """
    mom_tparams = OrderedDict()
    xi_tparams = OrderedDict()
    for k, p0 in tparams.iteritems():
        mom_tparams[k] = theano.shared(p0.get_value() * 0. + 1e-10, name='%s_mom'%k) 
        xi_tparams[k] = theano.shared(p0.get_value() * 0. + 1e-10, name='%s_xi'%k) 
    
    a = theano.shared(numpy_floatX(2.))
    m_org = theano.shared(numpy_floatX(5.))
    c = theano.shared(numpy_floatX(5.))
    sigma_p = theano.shared(numpy_floatX(10.))
    sigma_xi = theano.shared(numpy_floatX(0.001))
    gamma_xi = theano.shared(numpy_floatX(1))
    logger = logging.getLogger('eval_ptb_sgmgnht')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_ptb_sgmgnht.log')
    logger.info('a {} m {} c {} s_p{} s_xi{} g_xi{}'.format(a.get_value(), m_org.get_value(), c.get_value(), sigma_p.get_value(), sigma_xi.get_value(), gamma_xi.get_value()))
    
    p = tensor.vector('p', dtype='float32')
    
    """ default: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p0.get_value() * 0., name='%s_grad'%k) 
                for k, p0 in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
    # reset mom
    # counter = theano.shared(numpy_floatX(0.))
    # updates.append((counter,counter+1))
 
    for p, mom, xi, g in zip(tparams.values(),mom_tparams.values(),xi_tparams.values(), gshared):
	#rms prop
        t = theano.shared(p.get_value() * 0.)
        t_new = rho * t + (1-rho) * g**2
        updates.append((t, t_new))

        m = (tensor.sqrt(t_new) + 1e-10)
        m = m/tensor.max(m)*m_org
        #m = tensor.switch(tensor.ge(m,1*m_org), 1*m_org, m)
        m = tensor.switch(tensor.le(m,m_org*0.01), m_org*0.01, m)
        
        g_f = tensor.sgn(mom)/m*(tensor.abs_(mom)**(1/a))
        K_f = -g_f + 2/c*(c*g_f + tensor.log(1+tensor.exp(-c*g_f)))
        
        psi_f_1 = (1- tensor.exp(-c*g_f) )/( 1 + tensor.exp(-c*g_f) )
        f1_f_1 = 1/m/a*psi_f_1*(tensor.abs_(mom+1e-100)**(1/a-1))
        psi_grad_f_1 = 2*c*tensor.exp(- c*g_f)/(1 + tensor.exp(-c*g_f))**2
        f3_f_1 =  1/m**2/a**2*(psi_f_1**2-psi_grad_f_1)*tensor.abs_(mom+1e-100)**(2/a-2) - (1/a-1)/m/a*psi_f_1*tensor.sgn(mom)*tensor.abs_(mom+1e-100)**(1/a-2)
        
        psi_f = (tensor.exp(c*g_f) - 1)/(tensor.exp(c*g_f) + 1)
        f1_f = 1/m/a*psi_f*(tensor.abs_(mom+1e-100)**(1/a-1))
        psi_grad_f = 2*c*tensor.exp(c*g_f)/(tensor.exp(c*g_f) + 1)**2
        f3_f =  1/m**2/a**2*(psi_f**2-psi_grad_f)*tensor.abs_(mom+1e-100)**(2/a-2) - (1/a-1)/m/a*psi_f*tensor.sgn(mom)*tensor.abs_(mom+1e-100)**(1/a-2)
 
        temp_f1 = tensor.switch(tensor.ge(g_f,0), f1_f_1, f1_f)
        temp_f3 = tensor.switch(tensor.ge(g_f,0), f3_f_1, f3_f)       


        noise_p = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        noise_xi = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        
        #lr_new = 1 / tensor.sqrt(tensor.abs_(temp_f1)) * lr 
        lr_new = lr
        updated_p = p +  temp_f1 * lr_new
        #updated_mom = (mom - temp_f1* xi *lr  - g * lr * ntrain + tensor.sqrt(2*sigma_p*lr) * noise_p)* (1-tensor.eq(tensor.mod(iterations,100),0)) + randmg * tensor.eq(tensor.mod(iterations,100),0)
        updated_mom = mom - 1.2*temp_f1* xi *lr_new  - g * lr_new * ntrain + tensor.sqrt(2*sigma_p*lr_new) * noise_p
        updated_xi = xi + temp_f3* sigma_xi * lr_new - (xi - sigma_p)*gamma_xi*lr_new + tensor.sqrt(2*sigma_xi*gamma_xi*lr_new) * noise_xi 
        

        updates.append((p, updated_p))
            
        updates.append((mom, updated_mom))
        updates.append((xi, updated_xi))
    
    f_update = theano.function([lr,ntrain], [p,mom,m], updates=updates)
    
    return f_grad_shared, f_update
  

def SGMGHMC_old(tparams, cost, inps, ntrain, lr, iterations, rho=0.9, epsilon=1e-6, a_i = 2, clip_norm=5):
    """ Additional parameters """
    mom_tparams = OrderedDict()
    xi_tparams = OrderedDict()
    for k, p0 in tparams.iteritems():
        mom_tparams[k] = theano.shared(p0.get_value() * 0. + 1e-10, name='%s_mom'%k) 
        xi_tparams[k] = theano.shared(p0.get_value() * 0. + 1e-10, name='%s_xi'%k) 
    
    a = theano.shared(numpy_floatX(1.))
    m = theano.shared(numpy_floatX(1.))
    c = theano.shared(numpy_floatX(5.))
    sigma_p = theano.shared(numpy_floatX(10.))
    sigma_xi = theano.shared(numpy_floatX(1.))
    gamma_xi = theano.shared(numpy_floatX(0.001))
    logger = logging.getLogger('eval_ptb_sgmgnht')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_ptb_sgmgnht.log')
    logger.info('a {} m {} c {} s_p{} s_xi{} g_xi{}'.format(a.get_value(), m.get_value(), c.get_value(), sigma_p.get_value(), sigma_xi.get_value(), gamma_xi.get_value()))
    
    p = tensor.vector('p', dtype='float32')
    
    """ default: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p0.get_value() * 0., name='%s_grad'%k) 
                for k, p0 in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
 
    for p, mom, xi, g in zip(tparams.values(),mom_tparams.values(),xi_tparams.values(), gshared):
        
        g_f = tensor.sgn(mom)/m*(tensor.abs_(mom)**(1/a))
        K_f = -g_f + 2/c*(c*g_f + tensor.log(1+tensor.exp(-c*g_f)))
        
        psi_f_1 = (1- tensor.exp(-c*g_f) )/( 1 + tensor.exp(-c*g_f) )
        f1_f_1 = 1/m/a*psi_f_1*(tensor.abs_(mom+1e-100)**(1/a-1))
        psi_grad_f_1 = 2*c*tensor.exp(- c*g_f)/(1 + tensor.exp(-c*g_f))**2
        f3_f_1 =  1/m**2/a**2*(psi_f_1**2-psi_grad_f_1)*tensor.abs_(mom+1e-100)**(2/a-2) - (1/a-1)/m/a*psi_f_1*tensor.sgn(mom)*tensor.abs_(mom+1e-100)**(1/a-2)
        
        psi_f = (tensor.exp(c*g_f) - 1)/(tensor.exp(c*g_f) + 1)
        f1_f = 1/m/a*psi_f*(tensor.abs_(mom+1e-100)**(1/a-1))
        psi_grad_f = 2*c*tensor.exp(c*g_f)/(tensor.exp(c*g_f) + 1)**2
        f3_f =  1/m**2/a**2*(psi_f**2-psi_grad_f)*tensor.abs_(mom+1e-100)**(2/a-2) - (1/a-1)/m/a*psi_f*tensor.sgn(mom)*tensor.abs_(mom+1e-100)**(1/a-2)
 
        temp_f1 = tensor.switch(tensor.ge(g_f,0), f1_f_1, f1_f)
        temp_f3 = tensor.switch(tensor.ge(g_f,0), f3_f_1, f3_f)       


        noise_p = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        noise_xi = trng.normal(p.get_value().shape, avg = 0.0, std = 1., 
                          dtype='float32')
        # generata gamma(a,2): N(0,1)^2 = gamma(1/2,2)
        noise_temp = tensor.zeros(p.get_value().shape)
        for aa in xrange(a_i*2):
            this_noise = trng.normal(p.get_value().shape, avg = 0.0, std = 1., dtype='float32')
            noise_temp = tensor.inc_subtensor(noise_temp[:], this_noise**2)
        randmg = (noise_temp*m/2)**a*tensor.sgn(trng.normal(p.get_value().shape, avg = 0.0, std = 1., dtype='float32'))    

        
        updated_p = p +  temp_f1 * lr
	updated_mom = (mom - temp_f1* xi *lr  - g * lr * ntrain + tensor.sqrt(2*sigma_p*lr) * noise_p)* (1-tensor.eq(tensor.mod(iterations,100),0)) + randmg * tensor.eq(tensor.mod(iterations,100),0)
        #updated_mom = mom - temp_f1* xi *lr  - g * lr * ntrain + tensor.sqrt(2*sigma_p*lr) * noise_p
        temp_xi = trng.normal(p.get_value().shape, avg = sigma_p, std = tensor.sqrt(sigma_xi/2) , dtype='float32')
        updated_xi = (xi + temp_f3* sigma_xi * lr - (xi - sigma_p)*gamma_xi*lr + tensor.sqrt(2*sigma_xi*gamma_xi*lr) * noise_xi) * (1-tensor.eq(tensor.mod(iterations,100),50)) + temp_xi * tensor.eq(tensor.mod(iterations,100),50)
        

        updates.append((p, updated_p))
        updates.append((mom, updated_mom))
        updates.append((xi, updated_xi))
    
    f_update = theano.function([lr,ntrain,iterations], [p,mom,xi], updates=updates)
    #f_params = theano.function([], [a, m, c, mom.shape])
    return f_grad_shared, f_update
  
def SGLD(tparams, cost, inps, ntrain, lr, clip_norm=5):
    """ default: lr=0.01 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):  
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype='float32')
        updated_p = p - lr * g + tensor.sqrt(2.*lr/ntrain) * eps
        updates.append((p, updated_p))
    
    f_update = theano.function([lr,ntrain], [], updates=updates)
    
    return f_grad_shared, f_update 
