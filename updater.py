# -*- coding: utf-8 -*-
#

import chainer
import chainer.functions as F
import numpy as np

class PretrainUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.wae = kwargs.pop('model')
        self.wae_lambda = 100.
        super().__init__(*args, **kwargs)

    def update_core(self):
        opt = self.get_optimizer('pretrain')
        xp = self.wae.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        batch = xp.array(batch)
        batch_images = chainer.Variable(batch)
        batch_noise = chainer.Variable(self.wae.sample_pz(batchsize))
        sample_qz = self.wae.enc(batch_images)
        sample_pz = batch_noise

        mean_pz = F.mean(sample_pz, axis=0, keepdims=True)
        mean_qz = F.mean(sample_qz, axis=0, keepdims=True)
        mean_loss = F.mean(F.square(mean_pz - mean_qz))

        mpz = F.broadcast_to(mean_pz, sample_pz.shape)
        cov_pz = F.matmul((sample_pz - mpz),  (sample_pz - mpz), transa=True)
        cov_pz /= batchsize - 1.
        mqz = F.broadcast_to(mean_qz, sample_qz.shape)
        cov_qz = F.matmul((sample_qz - mqz), (sample_qz - mqz), transa=True)
        cov_qz /= batchsize - 1.
        cov_loss = F.mean(F.square(cov_pz - cov_qz))
        loss = mean_loss + cov_loss

        self.wae.enc.cleargrads()
        loss.backward()
        opt.update()
        chainer.reporter.report({'loss_pre': loss})

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.wae = kwargs.pop('model')
        self.wae_lambda = 100.
        super().__init__(*args, **kwargs)

    def update_core(self):
        ae_opt = self.get_optimizer('autoencoder')
        xp = self.wae.xp
        wae = self.wae

        batch = self.get_iterator('main').next()
        batch = xp.array(batch, dtype=xp.float32)
        batchsize = len(batch)

        batch_images = chainer.Variable(batch)
        batch_noise = chainer.Variable(wae.sample_pz(batchsize))

        # calc penalty
        sample_qz = wae.enc(batch_images)
        sample_pz = batch_noise
        penalty = wae.mmd_penalty(sample_qz, sample_pz)

        # calc reconstruct loss
        real = batch_images
        reconstr = wae.dec(wae.enc(batch_images))
        loss_reconstruct = wae.reconstruction_loss(real, reconstr)

        # calc objective
        wae_objective = loss_reconstruct + \
                        penalty * self.wae_lambda

        wae.enc.cleargrads()
        wae.dec.cleargrads()
        wae_objective.backward()
        ae_opt.update()

        chainer.reporter.report({'loss_recon': loss_reconstruct})
        chainer.reporter.report({'penalty': penalty})
        chainer.reporter.report({'wae_obj': wae_objective})
        
