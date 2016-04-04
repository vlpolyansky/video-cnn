import os, sys
import random as rand


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def randfloat(a, b):
    return rand.random() * (b - a) + a


def my_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()
