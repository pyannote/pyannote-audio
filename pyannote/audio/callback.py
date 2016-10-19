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
# Hervé BREDIN - http://herve.niderb.fr


import matplotlib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import os.path
import datetime
import numpy as np
from keras.callbacks import Callback


class LoggingCallback(Callback):
    """Logging callback

    Parameters
    ----------
    log_dir : str
    log : list of tuples, optional
        Defaults to [('train', 'loss')]
    extract_embedding : func
        Function that takes Keras model as input and returns the actual model.
        This is useful for embedding that are not directly optimized by Keras.
        Defaults to identity function.
    """

    def __init__(self, log_dir, log=None, extract_embedding=None):
        super(LoggingCallback, self).__init__()

        # make sure path is absolute
        self.log_dir = os.path.realpath(log_dir)

        if extract_embedding is None:
            extract_embedding = lambda model: model
        self.extract_embedding = extract_embedding

        # create log_dir directory (and subdirectory)
        os.makedirs(self.log_dir)
        os.makedirs(self.log_dir + '/weights')

        if log is None:
            log = [('train', 'loss')]
        self.log = log

        self.values = {}
        for subset, name in self.log:
            if subset not in self.values:
                self.values[subset] = {}
            if name not in self.values[subset]:
                self.values[subset][name] = []

    def get_loss(self, epoch, subset, logs={}):
        if subset != 'train':
            raise ValueError('"loss" is only available for "train"')
        value = logs['loss']
        minimize = True
        return value, minimize

    def get_accuracy(self, epoch, subset, logs={}):
        if subset != 'train':
            raise ValueError('"accuracy" is only available for "train"')
        value = logs['acc']
        minimize = False
        return value, minimize

    def get_value(self, epoch, name, subset, logs={}):
        get_value = getattr(self, 'get_' + name)
        value, minimize = get_value(epoch, subset, logs=logs)
        return value, minimize

    def on_epoch_begin(self, epoch, logs={}):
        """Save architecture before first epoch"""

        if epoch > 0:
            return

        architecture = self.log_dir + '/architecture.yml'
        model = self.extract_embedding(self.model)
        yaml_string = model.to_yaml()
        with open(architecture, 'w') as fp:
            fp.write(yaml_string)

    def on_epoch_end(self, epoch, logs={}):
        """Save weights (and various curves) after each epoch"""

        # keep track of when the epoch ended
        now = datetime.datetime.now().isoformat()

        # save model after this epoch
        PATH = self.log_dir + '/weights/{epoch:04d}.h5'
        current_weights = PATH.format(epoch=epoch)

        # save current weights
        try:
            model = self.extract_embedding(self.model)
            model.save_weights(current_weights, overwrite=True)
        except Exception as e:
            pass

        for subset, name in self.log:

            value, minimize = self.get_value(epoch, name, subset, logs=logs)

            # keep track of value after last epoch
            self.values[subset][name].append(value)
            values = self.values[subset][name]

            # write value to file
            PATH = self.log_dir + '/{name}.{subset}.txt'
            path = PATH.format(subset=subset, name=name)
            TXT_TEMPLATE = '{epoch:d} ' + now + ' {value:.3f}\n'

            mode = 'a' if epoch > 0 else 'w'
            try:
                with open(path, mode) as fp:
                    fp.write(TXT_TEMPLATE.format(epoch=epoch, value=value))
                    fp.flush()
            except Exception as e:
                pass

            # keep track of 'best value' model
            if minimize:
                best_epoch = np.argmin(values)
                best_value = np.min(values)
            else:
                best_epoch = np.argmax(values)
                best_value = np.max(values)

            if best_epoch == epoch:
                LINK_NAME = self.log_dir + '/best.{name}.{subset}.h5'
                link_name = LINK_NAME.format(subset=subset, name=name)

                try:
                    os.remove(link_name)
                except Exception as e:
                    pass

                try:
                    os.symlink(current_weights, link_name)
                except Exception as e:
                    pass

            # plot values to file and mark best value so far
            plt.plot(values, 'b')
            plt.plot([best_epoch], [best_value], 'bo')

            plt.xlabel('epoch')

            YLABEL = '{name} on {subset}'
            ylabel = YLABEL.format(name=name, subset=subset)
            plt.ylabel(ylabel)

            TITLE = '{name} = {best_value:.3f} on {subset} @ epoch #{best_epoch:d}'
            title = TITLE.format(name=name, subset=subset, best_value=best_value, best_epoch=best_epoch)
            plt.title(title)

            plt.tight_layout()

            # save plot as PNG
            PATH = self.log_dir + '/{name}.{subset}.png'
            path = PATH.format(subset=subset, name=name)
            try:
                plt.savefig(path, dpi=150)
            except Exception as e:
                pass

            # save plot as EPS
            PATH = self.log_dir + '/{name}.{subset}.eps'
            path = PATH.format(subset=subset, name=name)
            try:
                plt.savefig(path)
            except Exception as e:
                pass

            plt.close()
