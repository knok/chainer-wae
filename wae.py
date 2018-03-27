# -*- coding: utf-8 -*-
#

import os
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class Decoder(chainer.Chain):
    def __init__(self, zdim, units, layers, wscale=0.02, ksize=4,
                 output_shape=(3, 64, 64)):
        self.zdim = zdim
        self.units = units
        self.layers = layers
        self.wscale = wscale
        self.ksize = ksize
        self.output_shape = output_shape
        height = output_shape[1] // 2 ** layers
        width = output_shape[2] // 2 ** layers
        super().__init__()
        w = chainer.initializers.Normal(wscale)
        prev = zdim
        with self.init_scope():
            self.linear = L.Linear(zdim, units * height * width)
            prev = units
            for i in range(layers - 1):
                scale = 2 ** (i + 1)
                _out_shape = units // scale
                mname = "deconv%d" % i
                setattr(self, mname,
                        L.Deconvolution2D(prev, _out_shape, ksize=ksize,
                                          stride=2, pad=1, initialW=w))
                mname = "bn%d" % i
                setattr(self, mname, L.BatchNormalization(_out_shape))
                prev = _out_shape
            self.final = L.Deconvolution2D(prev, 3, ksize=ksize,
                                           stride=2, pad=1, initialW=w)

    def predict(self, x):
        bsize = x.shape[0]
        height = self.output_shape[1] // 2 ** self.layers
        width = self.output_shape[2] // 2 ** self.layers
        h = self.linear(x)
        h = F.reshape(h, (bsize, self.units, width, height))
        h = F.relu(h)
        for i in range(self.layers - 1):
            deconv = "deconv%d" % i
            bn = "bn%d" % i
            h = self[deconv](h)
            h = self[bn](h)
            h = F.relu(h)
        h = self.final(h)
        ret = F.tanh(h)
        return ret
                                                       
            
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
    d = Decoder(2, 100, 3)
    t = d.predict(y)
    import pdb; pdb.set_trace()
    
