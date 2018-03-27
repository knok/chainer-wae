# -*- coding: utf-8 -*-
#

import os
import argparse
import numpy as np
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions
import matplotlib
import matplotlib.pyplot as plt

import wae
import data
import updater

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', required=True)
    p.add_argument('--zdim', '-z', type=int, default=2)
    p.add_argument('--out', '-o', type=str, default="result")
    p.add_argument('--units', '-u', type=int, default=100)
    p.add_argument('--layers', '-l', type=int, default=3)
    p.add_argument('--batchsize', '-b', type=int, default=32)
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--epoch', '-e', type=int, default=1000)
    args = p.parse_args()
    return args

def save_graph():
    @training.make_extension(trigger=(100, 'epoch'))

    def _save_graph(trainer):
        iter = trainer.updater.get_iterator('main')
        batch = iter.next()
        bsize = len(batch)
        if bsize > 10:
            batch = batch[:10]
            bsize = 10
        model = trainer.updater.wae
        xp = model.xp
        batch = xp.array(batch)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            recon = model.dec(model.enc(batch))
            recon = recon.data
        def make_himg(img):
            img = img.transpose(0, 2, 3, 1)
            img = xp.hstack(img)
            img = (img + 1.0) / 2
            img = chainer.cuda.to_cpu(img)
            return img
        img1 = make_himg(batch)
        img2 = make_himg(recon)
        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(2, 1)
        for i, (img, title) in enumerate(
                zip([img1, img2], ["input", "reconstruct"])):
            plt.subplot(gs[i, 0])
            ax = plt.imshow(img)
            ax = plt.title(title)
        fname = "fig_{.updater.iteration}.png".format(trainer)
        fname = os.path.join(trainer.out, fname)
        plt.savefig(fname)
        plt.close()
       
    return _save_graph

def main():
    args = get_args()
    model = wae.WAE(args.zdim, args.units, args.layers)
    if args.gpu >= 0:
        model.to_gpu()
    dataset = data.Data64(args.data)

    # opimizer
    pre_opt = chainer.optimizers.Adam()
    pre_opt.setup(model.enc)
    updater_args = {
        "device": args.gpu
    }
    ae_opt = chainer.optimizers.Adam()
    ae_opt.setup(model)
    opts = {}
    opts["pretrain"] = pre_opt
    opts["autoencoder"] = ae_opt
    updater_args["optimizer"] = opts
    updater_args["model"] = model

    step_max = 200
    pretrain_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)
    updater_args["iterator"] = {"main": pretrain_iter}
    pre_updater = updater.PretrainUpdater(**updater_args)
    trainer = training.Trainer(pre_updater, (step_max, 'iteration'),
                                   out=args.out)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PrintReport(["loss_pre"]),
                   trigger=(50, 'iteration'))

    trainer.run()

    del trainer
    del pre_updater
    del pretrain_iter
    del pre_opt

    max_epoch = args.epoch
    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)
    updater_args["iterator"] = {"main": train_iter}
    main_updater = updater.Updater(**updater_args)
    trainer = training.Trainer(main_updater, (max_epoch, 'epoch'),
                               out=args.out)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PrintReport(["loss_recon", "penalty", "wae_obj"]),
                   trigger=(50, 'iteration'))
    trainer.extend(extensions.snapshot(), trigger=(100, 'epoch'))
    trainer.extend(save_graph())
    trainer.run()

    chainer.serializers.save_npz(args.out+ "/wae.npz", model)

if __name__ == '__main__':
    main()
