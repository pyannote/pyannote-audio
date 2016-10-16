#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Gregory GELLY - gregory.gelly@limsi.fr

from keras.optimizers import Optimizer 
import keras.backend as K

class SMORMS3(Optimizer):
    '''SMORMS3 optimizer.
    Default parameters follow those provided in the blog.
    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [RMSprop loses to SMORMS3 - Beware the Epsilon!](http://sifter.org/~simon/journal/20150420.html)
    '''
    def __init__(self, lr=0.001, epsilon=1e-16, decay=0., **kwargs):
        super(SMORMS3, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.decay = K.variable(decay)
        self.inital_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        mems = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs + mems

        for p, g, m, v, mem in zip(params, grads, ms, vs, mems):
            r = 1. / (1. + mem)
            m_t = (1. - r) * m + r * g
            v_t = (1. - r) * v + r * K.square(g)
            denoise = K.square(m_t) / (v_t + self.epsilon)
            p_t = p - g * K.minimum(lr, denoise) / (K.sqrt(v_t) + self.epsilon)
            mem_t = 1. + mem * (1. - denoise)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(mem, mem_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(SMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SSMORMS3(Optimizer):
    '''SSMORMS3 optimizer for Stabilized SMORMS3.
    Slight modification of SMORMS3 that stabilizes its behavior
    # Arguments
        epsilon: float >= 0. Fuzz factor.
    # References
        - [RMSprop loses to SMORMS3 - Beware the Epsilon!](http://sifter.org/~simon/journal/20150420.html)
    '''
    def __init__(self, epsilon=1e-8, **kwargs):
        super(SSMORMS3, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        mems = [K.zeros(shape) for shape in shapes]
        denoises = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs + mems + denoises

        for p, g, m, v, mem, denoise in zip(params, grads, ms, vs, mems, denoises):
            r = K.minimum(0.2, K.maximum(0.005, 1. / (1. + mem)))
            mem_t = 1. / r - 1.
            m_t = (1. - r) * m + r * g
            v_t = (1. - r) * v + r * K.square(g)
            denoise_t = 0.99 * denoise + 0.01 * K.square(m_t) / (v_t + self.epsilon)
            p_t = p - g * denoise_t / (K.sqrt(v_t) + self.epsilon)
            mem_t = K.maximum(0., 1. + mem_t * (1. - denoise_t))

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(mem, mem_t))
            self.updates.append(K.update(denoise, denoise_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'epsilon': self.epsilon}
        base_config = super(SSMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

