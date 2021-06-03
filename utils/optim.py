# -*- coding: utf-8 -*-

import math
import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr, decay_step = 1000, 
                       decay_rate=0.9, steps=0):
        self.init_lr = lr
        self.steps = steps
        self._optimizer = optimizer
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def step(self):
        '''Step with the inner optimizer'''
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.steps += 1
        if self.steps >= self.decay_step:
            lr = self.init_lr * math.pow(self.decay_rate, 
                                         int(self.steps / self.decay_step))
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self.init_lr


