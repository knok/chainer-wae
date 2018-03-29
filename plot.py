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
    p.add_argument('--data', '-d', required=True)
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
    dataset = data.Data64(args.data)
    if args.gpu >= 0:
        model.to_gpu()

    chainer.serializers.load_npz(args.model, model)

    data_iter = chainer.iterators.SerialIterator(dataset, args.batchsize,
                                                 repeat=False, shuffle=False)
    vectors = []
    with chainer.using_config('train', False):
        for d in data_iter:
            d = np.array(d)
            x = chainer.Variable(d)
            sample_vec = model.enc(d)
            if args.gpu >=0:
                sample_vec = chainer.cuda.to_cpu(sample_vec)
            vectors.append(sample_vec.data)
    z = np.vstack(vectors)
    # import pdb; pdb.set_trace()
    # pass

    plt.scatter(z[:, 0], z[:, 1])
    plt.show()

if __name__ == '__main__':
    main()
