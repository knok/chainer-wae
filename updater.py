# -*- coding: utf-8 -*-
#

import chainer
import numpy as np

class PretrainUpdater(chainer.training.StandardUpdater):
    def __init__(*args, **kwargs):
        self.wae = kwargs.pop['model']
        self.wae_lambda = 100.
        super().__init__(*args, **kwargs)

    def update_core(self):
        opt = self.get_optmizer('pretrain')
        xp = self.wae.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        batch_images = chainer.Variable(batch)
        batch_noise = chainer.Variable(wae.sample_pz(batchsize))
        sample_qz = self.wae.enc(batch_images)
        sample_pz = batch_noise
        
        mean_pz = xp.mean(sample_pz, axis=0)
        mean_qz = xp.mean(sample_qz, axis=0)
        mean_loss = xp.mean(xp.square(mean_pz - mean_qz))

        cov_pz = (sample_pz - mean_pz) * (sample_pz - mean_pz).T
        cov_pz /= batchsize - 1.
        cov_qz = (sample_qz - mean_qz) * (sample_qz - mean_qz).T
        cov_qz /= batchsize - 1.
        conv_loss = xp.mean(xp.square(conv_pz - cov_qz))
        loss = mean_loss + cov_loss

        wae.enc.cleargrads()
        loss.backward()
        opt.update()

class Updater(chainer.training.StandardUpdater):
    def __init__(*args, **kwargs):
        self.wae = kwargs.pop['model']
        self.wae_lambda = 100.
        super().__init__(*args, **kwargs)

    def update_core(self):
        ae_opt = self.get_optmizer('autoencodder')
        xp = self.wae.xp

        batch = self.get_iterator('main').next()
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
        
