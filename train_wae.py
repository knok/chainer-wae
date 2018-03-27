# -*- coding: utf-8 -*-
#

import os
import argparse
import numpy as np
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

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
    args = p.parse_args()
    return args

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

    max_epoch = 1000
    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)
    updater_args["iterator"] = {"main": train_iter}
    main_updater = updater.Updater(**updater_args)
    trainer = training.Trainer(main_updater, (max_epoch, 'epoch'),
                               out=args.out)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PrintReport(["loss_recon", "penalty", "wae_obj"]),
                   trigger=(50, 'iteration'))
    trainer.run()
                   

if __name__ == '__main__':
    main()
