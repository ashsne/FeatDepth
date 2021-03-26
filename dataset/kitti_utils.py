#!/home/ash/anaconda3/envs/pytorch/bin/python

import torch
import numpy as np
import cv2


def readlines(filename):
    """Read all the lines from a file and return a list
    """
    with open(filename, r) as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    pass



if __name__ == '__main__':
    print('Hello')

