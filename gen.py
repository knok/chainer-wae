# -*- coding: utf-8 -*-
#

import os
import argparse
import numpy as np
import chainer
import matplotlib.pyplot as plt

import wae
import data

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--zdim', '-z', type=int, default=2)
    p.add_argument('--out', '-o', type=str, default="result")
    p.add_argument('--units', '-u', type=int, default=100)
    p.add_argument('--layers', '-l', type=int, default=3)
    p.add_argument('--batchsize', '-b', type=int, default=32)
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--model', '-m', required=True)
    args = p.parse_args()
    return args

def main():
    args = get_args()
    model = wae.WAE(args.zdim, args.units, args.layers)
    if args.gpu >= 0:
        model.to_gpu()

    chainer.serializers.load_npz(args.model, model)

    batch_noise = model.sample_pz(10)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        sample_gen = model.dec(batch_noise)

    if args.gpu >=0:
        sample_gen = chainer.cuda.to_cpu(sample_gen)

    img = sample_gen.data.transpose(0, 2, 3, 1)
    img = np.hstack(img)
    img = (img + 1.0) / 2
    plt.imshow(img)
    plt.savefig('img.png')

if __name__ == '__main__':
    main()
