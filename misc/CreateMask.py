import random

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
test_img = cv2.imread(r'/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/train_v2/000194a2d.jpg')

encoded_pixels = pd.read_csv(r'/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/train_ship_segmentations_v2.csv')

print(len(encoded_pixels.loc[np.isnan(3,'EncodedPixels')].split(' ')))
df = encoded_pixels.loc[(encoded_pixels['EncodedPixels']=='Nan')]
test = encoded_pixels.loc[3,'EncodedPixels'].split(' ')

#convert string to integers using map function
rle = list(map(int,test))

#First number is starting pixel and next number is count from that starting pixel. Lets
#bring them into 2 sep list
pixel, pixel_count = [],[]
[pixel.append(rle[i]) if i%2 == 0 else pixel_count.append(rle[i]) for i in range(0,len(rle))]

#Lets gen masked pixels locations where exactly the mask is there using above 2 lists
rle_pixels = [list(range(pixel[i],pixel[i]+pixel_count[i])) for i in range(0,len(pixel))]

#Now lets convert list of lists into a single list
rle_mask_pixels = sum(rle_pixels,[])
percentage = .1
k = int(len(rle_mask_pixels)*percentage //2)
random_sample = random.sample(range(len(rle_mask_pixels)),k)
new_mask_pixels = [rle_mask_pixels[i] for i in random_sample]

mask_img = np.zeros((768*768,1),dtype=int)
mask_img[rle_mask_pixels] = 255
mask_img1 = np.zeros((768*768,1),dtype=int)
mask_img1[new_mask_pixels] = 255
mask = np.reshape(mask_img,(test_img.shape[0],test_img.shape[1])).T
mask1 = np.reshape(mask_img1,(test_img.shape[0],test_img.shape[1])).T
plt.imshow(mask1)
plt.show()