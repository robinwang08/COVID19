import matplotlib.pyplot as plt
import nrrd
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import scipy
import os
import cv2
from glob import glob
import numpy as np
import os
from glob import glob

def plot_3d(image, savePath, threshold=-300):
    # Position the scan upright,
    # So the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces,norm, val = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    #plt.show()
    plt.savefig(savePath)
    plt.close('all')


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array((image > -320 ), dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image




def circular_structure(radius):
    size = radius * 2 + 1
    size = np.float64(size)
    i, j = np.mgrid[0:size, 0:size]
    i -= (size / 2)
    j -= (size / 2)
    return np.sqrt(i ** 2 + j ** 2) <= radius


'''
# Actual lung segmentation code

patho = '/media/user1/Elements/Penn usable'
basedir = os.path.normpath(patho)
files = glob(basedir+'/*/*.nrrd')
to_path = '/home/user1/4TBHD/COR19_without_seg/add_seg'
for file in files:
    name = file.split('/')
    name = name[-1].split('.')
    name = name[0]
    to_file = to_path+'/'+name+'.npy'
    if os.path.exists(to_file):
        continue
    image,_ = nrrd.read(file)
    print(image.shape)

    segmented_lungs = segment_lung_mask(image, False)
    np.save(to_path+'/'+name,segmented_lungs)
    print(segmented_lungs.shape)
'''



#Visualization
#patho = '/home/user1/4TBHD/COR19/COR19_new/data'
patho = '/media/user1/new_seg'
save =  '/media/user1/check_seg/'
basedir = os.path.normpath(patho)
files = glob(basedir+'/*.npy')
for file in files:
    image = np.load(file)
    print('Processing file: ' + file)
    name = file.split('/')
    name = name[-1].split('.')
    name = name[0]
    savePath = save + name + '.jpg'
    if os.path.isfile(savePath):
        print(savePath + ' already exists')
        continue
    try:
        plot_3d(image, savePath, 0)
    except:
        print('Error with ' + savePath)
