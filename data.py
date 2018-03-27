# -*- coding: utf-8 -*-
#

import chainer
import numpy as np
from PIL import Image
import os

class Data64(chainer.dataset.dataset_mixin.DatasetMixin):
    def __init__(self, datadir):
        self.datadir = datadir
        _files = []
        for root, dirs, files in os.walk(datadir):
            for fname in files:
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    _files.append(os.path.join(root, fname))
        self.files = _files

    def __len__(self):
        return len(self.files)

    def load_image(self, i):
        fname = self.files[i]
        img = Image.open(fname)
        img = img.resize((64, 64))
        img = np.asarray(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = img / 127.5 - 1.0
        return img

    def get_example(self, i):
        return self.load_image(i)

