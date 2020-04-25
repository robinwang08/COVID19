import numpy as np
import nrrd
from glob import glob
import os

patho = '/media/user1/seg'
pathnew = '/media/user1/new_seg'
basedir = os.path.normpath(patho)
files = glob(basedir+'/*.npy')

for file in files:
    name = file.split('/')
    name = name[-1].split('.')
    name = name[0]
    mask = np.load(file)
    savePath = pathnew + '/' + name + '.nrrd'
    nrrd.write(savePath, mask)