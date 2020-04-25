import nrrd
import numpy as np
from glob import glob
import os

patho = '/media/user1/seg'
pathnew = '/media/user1/new_seg'
basedir = os.path.normpath(patho)
files = glob(basedir+'/*.nrrd')

for file in files:
    name = file.split('/')
    name = name[-1].split('.')
    name = name[0]
    mask,_ = nrrd.read(file)
    np.save(pathnew+'/'+name,mask)