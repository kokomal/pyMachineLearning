# coding:utf-8

import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
