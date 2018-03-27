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
        batch = np.array(batch)
        batch_images = chainer.Variable(batch)
        batch_noise = chainer.Variable(self.wae.sample_pz(batchsize))
        sample_qz = self.wae.enc(batch_images)
        sample_pz = batch_noise

        mean_pz = xp.mean(sample_pz, axis=0)
        mean_pz = F.hstack(mean_pz)
        mean_pz = F.reshape(mean_pz, (1, mean_pz.shape[0]))
        mean_qz = xp.mean(sample_qz, axis=0)
        mean_qz = F.hstack(mean_qz)
        mean_qz = F.reshape(mean_qz, (1, mean_qz.shape[0]))
        mean_loss = xp.square(mean_pz - mean_qz)
        x = 1
        for y in mean_loss.shape:
            x *= y
        mean_loss = xp.sum(mean_loss) / x
        #import pdb; pdb.set_trace()

        mpz = xp.broadcast_to(mean_pz.data, sample_pz.shape)
        cov_pz = xp.dot((sample_pz - mpz).T,  (sample_pz - mpz))
        cov_pz /= batchsize - 1.
        mqz = xp.broadcast_to(mean_qz.data, sample_qz.shape)
        cov_qz = xp.dot((sample_qz - mqz).T, (sample_qz - mqz))
        cov_qz /= batchsize - 1.
        cov_loss = xp.square(cov_pz - cov_qz)
        l = 1
        for x in cov_loss.shape:
            l += x
        cov_loss = cov_loss.sum() / x
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

        batch = self.get_iterator('main').next()
        batch = np.array(batch, dtype=np.float32)
        batchsize = len(batch)

        batch_images = chainer.Variable(batch)
        batch_noise = chainer.Variable(wae.sample_pz(batchsize))

        # calc penalty
        sample_qz = self.wae.enc(batch_images)
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
        
