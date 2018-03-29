# -*- coding: utf-8 -*-
#

import os
import argparse
import numpy as np
import chainer
import matplotlib.pyplot as plt
import matplotlib

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
    imgs = []
    vectors = []
    gen_imgs = []
    with chainer.using_config('train', False):
        for d in data_iter:
            d = np.array(d)
            imgs.append(d)
            x = chainer.Variable(d)
            sample_vec = model.enc(d)
            generated = model.dec(sample_vec)
            if args.gpu >=0:
                sample_vec = chainer.cuda.to_cpu(sample_vec)
                generated = chainer.cuda.to_cpu(generated)
            vectors.append(sample_vec.data)
            gen_imgs.append(generated.data)

    imgs = np.vstack(imgs).transpose(0, 2, 3, 1)
    imgs = imgs * 0.5 + 0.5
    z = np.vstack(vectors)
    gen_imgs = np.vstack(gen_imgs).transpose(0, 2, 3, 1)
    gen_imgs = gen_imgs * 0.5 + 0.5

    zz = []
    z_idx = []
    for i in range(z.shape[0]):
        t = 0.5
        if z[i, 0] < t and z[i, 0] > -t and z[i, 1] < t and z[i, 1] > -t:
            zz.append(z[i])
            z_idx.append(i)
    zz = np.array(zz)
    z_imgs = imgs[z_idx]
    gz_imgs = gen_imgs[z_idx]
    l = z_imgs.shape[0]
    cols = 8
    m = (cols - l % cols)
    filler = np.ones((m, z_imgs.shape[1], z_imgs.shape[2], z_imgs.shape[3]),
                     dtype=np.float32)
    z_imgs = np.vstack([z_imgs, filler])
    zi = np.concatenate(np.split(z_imgs, cols), axis=2)
    zi = np.concatenate(zi, axis=0)
    gz_imgs = np.vstack([gz_imgs, filler])
    gzi = np.concatenate(np.split(gz_imgs, cols), axis=2)
    gzi = np.concatenate(gzi, axis=0)
    #import pdb; pdb.set_trace()

    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax = plt.subplot(gs[:, 0])
    plt.title("latent space")
    plt.scatter(z[:, 0], z[:, 1])
    plt.scatter(zz[:, 0], zz[:, 1], color="red")
    ax = plt.subplot(gs[0, 1])
    plt.title("input images")
    plt.imshow(zi)
    ax = plt.subplot(gs[1, 1])
    plt.title("generated images")
    plt.imshow(gzi)
    plt.show()
    plt.close()
    
    # import pdb; pdb.set_trace()
    # pass

if __name__ == '__main__':
    main()
