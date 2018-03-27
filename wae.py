# -*- coding: utf-8 -*-
#

import os
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class Encoder(chainer.Chain):
    def __init__(self, zdim, units, layers, wscale=0.02, ksize=4):
        self.zdim = zdim
        self.units = units
        self.layers = layers
        self.wscale = wscale
        self.ksize = ksize
        super().__init__()
        w = chainer.initializers.Normal(wscale)
        with self.init_scope():
            prev = 3
            for i in range(layers):
                scale = 2 ** (layers - i - 1)
                mname = "conv%d" % i
                setattr(self, mname, L.Convolution2D(
                    prev, units // scale, ksize=ksize, stride=2, pad=1,
                    initialW=w))
                mname = "bn%d" % i
                setattr(self, mname, L.BatchNormalization(units // scale))
                prev = units // scale
            self.linear = L.Linear(None, zdim)

    def predict(self, x):
        h = x
        for i in range(self.layers):
            conv = "conv%d" % i
            bn = "bn%d" % i
            h = self[conv](h)
            h = self[bn](h)
            h = F.relu(h)
        res = self.linear(h)
        return res

if __name__ == '__main__':
    e = Encoder(2, 100, 3)
    img = np.zeros((1, 3, 32, 32), dtype=np.float32)
    x = chainer.Variable(img)
    y = e.predict(x)
    import pdb; pdb.set_trace()
    
