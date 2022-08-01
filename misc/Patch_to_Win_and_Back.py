import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""
Input images must be square.
Will need to fix down the line
"""

img = np.array(Image.open(r'/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/train_v2/0a0a623a0.jpg').resize((160,160)))
img = img[np.newaxis,:]

##########Numpy Implementation#############
def patch_creation(img,patch_size = (8,8)):
    """Creates patches then flattens HxW dimensions
    img_shape -> (#p,patch_height,patch_width,patch_dims)->(#p,n,patch_dims)"""
    patches = []
    stride = np.int(patch_size[0])
    for i in range(0, img.shape[1], stride):
        temp = img[0][i:i + stride, i:i + stride, :]
        patches.append([temp])
    patches = np.squeeze(np.array(patches))
    print(patches.shape)
    p,h,w,c = patches.shape
    num_patches = p
    flatten_HW = np.reshape(patches,(1,p*h*w,c))
    return flatten_HW,num_patches,patches


def img_reconstruction(patches,num_patches,img_shape,patch_size = (8,8)):
    rec = np.zeros(img_shape)
    print(rec.shape)
    patches = np.reshape(patches,(1,num_patches,patch_size[0],patch_size[1],3))
    stride = np.int(patch_size[0])
    count = 0
    for i in range(0,len(rec),stride):
        rec[0][i:i+stride,i:i+stride,:] = patches[0][count]
        count += 1
    return img


def window(patches,win_size=4):


    return window



patches,num_patches,non_flat = patch_creation(img)
new_img     = img_reconstruction(patches,num_patches,img_shape=img.shape)


