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

    def __call__(self, x):
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

    def __call__(self, x):
        h = x
        for i in range(self.layers):
            conv = "conv%d" % i
            bn = "bn%d" % i
            h = self[conv](h)
            h = self[bn](h)
            h = F.relu(h)
        res = self.linear(h)
        return res

class WAE(chainer.Chain):
    def __init__(self, zdim, units, layers, wscale=0.02, ksize=4,
                 output_shape=(3, 64, 64)):
        self.zdim = zdim
        self.units = units
        self.layers = layers
        self.wscale = wscale
        self.ksize = ksize
        super().__init__()
        with self.init_scope():
            self.enc = Encoder(zdim, units, layers, wscale, ksize)
            self.dec = Decoder(zdim, units, layers, wscale, ksize)

    def mmd_penalty(self, qz, pz):
        xp = self.xp
        sigma2_p = 1. ** 2
        n = len(qz)
        nf = float(n)
        # half_size = (n * n - n) / 2 # for RBF kernel

        norm_pz = F.sum(F.square(pz), axis=1, keepdims=True)
        dotprods_pz = F.matmul(pz, pz, transb=True)
        max_axis = norm_pz.shape[0]
        b_norm_pz = F.broadcast_to(norm_pz, (max_axis, max_axis))
        bt_norm_pz = F.broadcast_to(norm_pz.T, (max_axis, max_axis))
        distance_pz = b_norm_pz + bt_norm_pz - 2. * dotprods_pz

        norm_qz = F.sum(F.square(qz), axis=1, keepdims=True)
        dotprods_qz = F.matmul(qz, qz, transb=True)
        max_axis = norm_qz.shape[0]
        b_norm_qz = F.broadcast_to(norm_qz, (max_axis, max_axis))
        bt_norm_qz = F.broadcast_to(norm_qz.T, (max_axis, max_axis))
        distance_qz = b_norm_qz + bt_norm_qz - 2. * dotprods_qz

        dotprods = F.matmul(qz, pz, transb=True)
        distances = b_norm_qz + bt_norm_pz - 2. * dotprods

        # IMQ kernel
        Cbase = 2. * self.zdim * sigma2_p
        stat = 0
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distance_qz)
            res1 += C / (C + distance_pz)
            res1 = res1 * (1 - self.xp.eye(n))
            res1 = F.sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = F.sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat

    def reconstrution_loss(self, x, y):
        # l2sq: c(x,y) = ||x - y||_2^2
        loss = F.sum(F.square(x - y), axis=(1, 2, 3))
        loss = 0.05 * F.mean(loss)
        return loss

    def sample_pz(self, batchsize):
        # normal distribution
        mean = np.zeros(self.zdim, dtype=np.float32)
        cov = np.identity(self.zdim, dtype=np.float32)
        noise = np.random.multivariate_normal( # cupy don't have this
            mean, cov, batchsize).astype(np.float32)
        noise = self.xp.asarray(noise, dtype=self.xp.float32)
        return noise

if __name__ == '__main__':
    m = WAE(2, 100, 3)
    img = np.zeros((1, 3, 32, 32), dtype=np.float32)
    x = chainer.Variable(img)
    y = m.enc(x)
    t = m.dec(y)
    import pdb; pdb.set_trace()
    
