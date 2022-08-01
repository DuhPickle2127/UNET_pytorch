import os
import random

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import glob

import albumentations as A
EXTENSION = "jpg"
imlist = glob.glob(f"IR/*/*.{EXTENSION}")
filenamedict = {os.path.basename(f):f for f in imlist}
point_annotations = glob.glob("IR/masks/*/*.json")

def resizeWithPoints(im, points=None, tile=256, tensor=True, make_batch=False, reduction=1):
    xresize, yresize = (im.shape[1] // 32 // reduction) * 32, (im.shape[0] // 32 // reduction) * 32
    resizeTransforms = A.Compose([A.Resize(height=yresize, width=xresize)],
                                 keypoint_params=A.KeypointParams(format="xy"))

    # Apply the transform to only the image
    appliedTransform = resizeTransforms(image=im, keypoints=points)
    im = appliedTransform["image"]
    points = appliedTransform["keypoints"]

    # Turn into a batch of 1 if necessary
    permute_order = (2, 0, 1)
    if make_batch:
        permute_order = (0, 3, 1, 2)
        im = np.expand_dims(im, axis=0)

    # Convert image tiles into tensor-prepared format if necessary, otherwise keep in RGB format
    if tensor:
        im = torch.Tensor(im.copy()).permute(permute_order) / 255

    # Return tiles
    return im, points

class BoatsDataset(Dataset):

    def __init__(self,dataframe,root_dir,percentage,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the images
            transform (callable,optional): Optional transform to be applied on a sample
        """
        self.encoded_pixels = pd.read_csv(dataframe)
        self.root_dir = root_dir
        self.percentage = percentage
        self.transform = transform


    def __len__(self):
        return len(self.encoded_pixels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = os.path.join(self.root_dir,self.encoded_pixels.iloc[item,1])
        image_name = self.encoded_pixels.iloc[item,1]
        image = np.array(Image.open(image).convert("RGB"))
        pixels_data = self.encoded_pixels.loc[self.encoded_pixels['ImageId']==image_name]

        row_num = pixels_data.iloc[:, 0].tolist()

        mask_img = np.zeros((768 * 768, 1), dtype=np.float32)
        if len(pixels_data) > 1:
            for i in row_num:
                rle = list(map(int, pixels_data['EncodedPixels'][i].split(' ')))
                pixel, pixel_count = [], []
                [pixel.append(rle[i]) if i % 2 == 0 else pixel_count.append(rle[i]) for i in range(0, len(rle))]
                rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(0, len(pixel))]
                rle_mask_pixels = sum(rle_pixels, [])

                k = int(len(rle_mask_pixels) * self.percentage)
                random_sample = random.sample(range(len(rle_mask_pixels)), k)
                new_mask_pixels = [rle_mask_pixels[i] for i in random_sample]

                mask_img[new_mask_pixels] = 1



                mask = np.reshape(mask_img, (image.shape[0], image.shape[1])).T
        else:
            mask = np.zeros((image.shape[0],image.shape[1]),dtype=np.float32)



        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = image/255
        # pixel_indices = np.where(mask == 1)
        # indices = []
        # for i in range(0,len(pixel_indices[0])):
        #     indices.append([pixel_indices[0][i],pixel_indices[1][i]])
        # print(mask.shape,image.shape)
        return image,mask

class IRBoats(Dataset):
    def __init__(self,image_names,dataframe,root_dir,percentage,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the images
            transform (callable,optional): Optional transform to be applied on a sample
        """
        self.image_names = pd.read_csv(image_names)
        self.encoded_pixels = pd.read_csv(dataframe)
        self.root_dir = root_dir
        self.percentage = percentage
        self.transform = transform
        self.tally      = []


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        # print(self.encoded_pixels.iloc[item,5])
        # print(self.image_names.iloc[item,1])
        image = os.path.join(self.root_dir,self.image_names.iloc[item,1])

        # print(self.image_names[item])
        pixels_data = self.encoded_pixels.loc[self.encoded_pixels['Image_Name'] == self.image_names.iloc[item,1]]
        # print(pixels_data)
        # pixels = pixels_data[:,0].tolist()
        image = np.array(Image.open(image).convert("RGB"))
        class_X = np.array(pixels_data["Cx"]).reshape(-1,1)
        class_Y = np.array(pixels_data["Cy"]).reshape(-1,1)
        pixels = np.hstack((class_X,class_Y))
        # print(pixels)
        # print(pixels_data[:,4:])
        mask = np.zeros((image.shape[0],image.shape[1]), dtype=np.float32)
        # row_num = pixels_data.iloc[:, 0].tolist()
        # for i in range(len(row_num)):
        for j in pixels:
            # print(j[0],j[1])
            if j[0] == 0 and j[1] == 0:
                mask[j[1],j[0]] = 0
            else:
                mask[j[1], j[0]] = 255


        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]



        return image, mask


class IRBoatsDuplicates(Dataset):
    def __init__(self,dataframe,root_dir,percentage=None,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the images
            transform (callable,optional): Optional transform to be applied on a sample
        """
        self.encoded_pixels = pd.read_csv(dataframe)
        self.root_dir = root_dir
        self.percentage = percentage
        self.transform = transform
        self.tally      = []


    def __len__(self):
        return len(self.encoded_pixels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        # print(self.encoded_pixels.iloc[item,5])
        # print(self.encoded_pixels.iloc[item,1])
        image = os.path.join(self.root_dir,self.encoded_pixels.iloc[item,1])
        image_name = self.encoded_pixels.iloc[item,1]
        pixels_data = self.encoded_pixels.loc[self.encoded_pixels['Image_Name'] == image_name]
        # print(pixels_data)
        # pixels = pixels_data[:,0].tolist()
        image = np.array(Image.open(image).convert("RGB"))
        # print(pixels_data["Bx"])
        # back_X = np.array(pixels_data["Bx"]).reshape(-1,1)
        # back_Y = np.array(pixels_data["By"]).reshape(-1,1)
        class_X = np.array(pixels_data["Cx"]).reshape(-1,1)
        class_Y = np.array(pixels_data["Cy"]).reshape(-1,1)
        Classpixels = np.hstack((class_X,class_Y))
        # Backpixels = np.hstack((back_X,back_Y))
        # print(pixels)
        # print(pixels_data[:,4:])
        mask = np.zeros((image.shape[0],image.shape[1]), dtype=np.float32)
        # row_num = pixels_data.iloc[:, 0].tolist()
        # for i in range(len(row_num)):
        for j in Classpixels:
            # print(j[0],j[1])
            if j[0] == 0 and j[1] == 0:
                mask[j[1],j[0]] = 0
            else:
                mask[j[1], j[0]] = 255
        # for j in Backpixels:
        #     # print(j[0],j[1])
        #     if j[0] == 0 and j[1] == 0:
        #         mask[j[1],j[0]] = 0
        #     else:
        #         mask[j[1], j[0]] = 255


        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]





        return image, mask


# Dataset class to read in segmentations
class SegDataPoints(Dataset):
    # Initialize data
    def __init__(self, point_annotations, transforms=None, scale_to=(480, 640)):
        self.points = point_annotations
        self.images = [filenamedict[f"{os.path.splitext(os.path.basename(p))[0]}.jpg"] for p in self.points]
        self.transforms = transforms

    # Return the length of the dataset
    def __len__(self):
        return len(self.images)

    # Return an item from the dataset
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            item = idx.tolist()
        # Select the next image, read it in
        im = cv2.cvtColor(cv2.imread(self.images[idx]),cv2.COLOR_BGR2RGB)

        # Get filename, load points
        filename = self.points[idx]
        points = json.loads(open(filename).read().strip())["samples"]
        points = [[int(p[0] / 853 * 640), int(p[1] / 640 * 480), p[2]] for p in points]

        # Resize image and points
        im, coords = resizeWithPoints(im, points=[p[:2] for p in points], reduction=2, tensor=False)
        points = [[int(c[0]), int(c[1]), int(pol[2])] for c, pol in zip(coords, points)]

        # Create mask and populate
        mask = np.zeros((im.shape[0], im.shape[1]))
        eval_mask = np.zeros((im.shape[0], im.shape[1]))
        # for p in points:
        #     (x, y, pol) = p
        #     if pol == -1:
        #         mask[y, x] = [1.0, 0.0]
        #     else:
        #         mask[y, x] = [0.0, 1.0]
        #     eval_mask[y, x] = 1
        for p in points:
            # print(p)
            if p[2] == 1:
                mask[p[1],p[0]] = 255
        if self.transforms is not None:
            augmentations = self.transforms(image=im,mask=mask)
            im = augmentations["image"]
            mask = augmentations["mask"]


        # torchim, torchmask = torch.Tensor(im),torch.Tensor(mask)
        # torcheval = torch.Tensor(eval_mask)
        #
        # # Scale image out of [0, 1]
        # torchim = torchim / 255
        # # Change mask to be shape of (CLASS_NUM, TILE, TILE)
        # torchmask = torch.permute(torchmask, (2, 0, 1))

        # return torchim, torchmask, torcheval
        return im,mask
class EVAL(Dataset):
    def __init__(self,root_dir,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the images
            transform (callable,optional): Optional transform to be applied on a sample
        """

        self.root_dir = root_dir
        self.image_name = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = os.path.join(self.root_dir,self.image_name[item])
        image = np.array(Image.open(image).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image


# image = np.array(Image.open(r'/home/ethan/PycharmProjects/UNET_pytorch/IR/boatPhotos2/Image4204.jpg'))
#
# csv_file = pd.read_csv(r'/home/ethan/PycharmProjects/UNET_pytorch/IR_train1_csv')
# # #
# images = sorted(os.listdir(r'/home/ethan/PycharmProjects/UNET_pytorch/IR/CurrentAnnotImgs'))
import albumentations as A
from albumentations.pytorch import ToTensorV2
val_transform = A.Compose(
                            [
                                # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                A.Normalize(),
                                # ToTensorV2(),
                            ],
                        )

train_transform = A.Compose(
    [
        A.Resize(height=160, width=240),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.1),
        #     A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        #     max_pixel_value=255.0,
        # ),
        A.ToGray(p=1.0),
        A.InvertImg(p=.5),
        # ToTensorV2(),
    ],
)

dataframe = pd.read_csv(r'train_csv')
dataset = BoatsDataset(dataframe=r'train_csv',root_dir=r'ETHAN_DATA/train_v2',transform=train_transform,percentage=1)

image,mask = dataset[3]
# dataset = IRBoatsDuplicates(dataframe='/home/ethan/PycharmProjects/UNET_pytorch/IR_train1_csv',root_dir='/home/ethan/PycharmProjects/UNET_pytorch/IR/CurrentAnnotImgs',percentage=1,transform=val_transform)
# points = SegDataPoints(point_annotations=point_annotations)
# # # # # # # #
# # # # # # #
# image,mask = points[500]
# img = cv2.cvtColor(cv2.imread(r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/10x5/AIRBUS/34Mil/FocalandDice/34Mil_10/Input/original_276.png'),cv2.COLOR_BGR2RGB)
# pred = cv2.imread(r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/10x5/AIRBUS/34Mil/FocalandDice/34Mil_10/Pred/pred_276.png')
# #
# import matplotlib.pyplot as plt
# input_folder =  r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/TestImgs/squiggle/Focal+DICE/Input'
# output_folder = r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/TestImgs/squiggle/Focal+DICE/Pred'
# input_list = sorted(os.listdir(input_folder))
# output_list = sorted(os.listdir(output_folder))
# k = 0
# for i,j in zip(input_list,output_list):
#     img = cv2.imread(os.path.join(input_folder,i))
#     pred = cv2.imread(os.path.join(output_folder,j))
#
#     mask = pred[:,:,0]
# # # image_combined = overlay(img,mask)
#     fig,ax = plt.subplots()
#     ax.imshow(img)
#     ax.imshow(mask,cmap='Blues',alpha=.4)
#     # for line in ax.lines:
#         # line.set_marker(None)
#
#     plt.tick_params(left = False, right = False , labelleft = False ,
#                     labelbottom = False, bottom = False)
#     plt.subplots_adjust(left=0.01, right=.99, top=0.999, bottom=0.0001)
#     plt.tight_layout()
#     fig.savefig(f'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/TestImgs/squiggle/Focal+DICE/plot/plot{k}.png')
#     k+=1
#     fig.show()

# test = images[100]
# check = csv_file.loc[csv_file["Image_Name"]== test]
# pixels = csv_file.iloc[0][1] == images[100]

# pixels_data = csv_file.loc[csv_file['Image_Name'] == 'Image11145.jpg']
# csv_file['EncodedPixels'] = csv_file['EncodedPixels'].fillna('0')
# k = 3
# image_name = csv_file.iloc[k,1]
# pixels_data = csv_file.loc[csv_file['ImageId'] == image_name]
# row_num = pixels_data.iloc[:,0].tolist()
# pixels = []
# mask_img = np.zeros((768 * 768, 1), dtype=int)
# for i in row_num:
#     rle = list(map(int,pixels_data['EncodedPixels'][i].split(' ')))
#     pixel, pixel_count = [], []
#     [pixel.append(rle[i]) if i % 2 == 0 else pixel_count.append(rle[i]) for i in range(0, len(rle))]
#     rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(0, len(pixel))]
#     rle_mask_pixels = sum(rle_pixels, [])
#     percentage = .1
#     k = int(len(rle_mask_pixels) * percentage // 2)
#     random_sample = random.sample(range(len(rle_mask_pixels)), k)
#     new_mask_pixels = [rle_mask_pixels[i] for i in random_sample]
#
#     mask_img[new_mask_pixels] = 255
#
# mask = np.reshape(mask_img,(768,768)).T
# plt.imshow(mask)
# plt.show()

# dataset = BoatsDataset(dataframe= csv_file,
#                        root_dir='/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/train_v2',
#                        percentage=1)
# #
# # dataloader = DataLoader(dataset,batch_size = 16,shuffle=True,num_workers=2)
# # #Test plot
# sample = dataset[15]
# mask = sample[1]
#
# indices = sample[2]
# plt.imshow(mask)
# plt.show()
