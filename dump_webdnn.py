# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import chainer

import wae

from webdnn.backend import generate_descriptor
from webdnn.frontend.chainer import ChainerConverter

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--zdim', '-z', type=int, default=2)
    p.add_argument('--units', '-u', type=int, default=100)
    p.add_argument('--layers', '-l', type=int, default=3)
    p.add_argument('--model', '-m', type=str, required=True)
    p.add_argument('--out', '-o', type=str, default="webdnn")
    args = p.parse_args()
    return args

def main():
    args = get_args()
    model = wae.WAE(args.zdim, args.units, args.layers)
    chainer.serializers.load_npz(args.model, model)

    with chainer.using_config('train', False):
        input = np.zeros((1, args.zdim), dtype=np.float32)
        x = chainer.Variable(input)
        y = model.dec(x)
    graph = ChainerConverter().convert([x], [y])
    exec_info = generate_descriptor("webgl", graph)
    exec_info.save(args.out)

if __name__ == '__main__':
    main()
