import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

img = np.array(Image.open(r'/home/ethan/PycharmProjects/UNET_pytorch/IR/boatPhotos4/Image11153.jpg'))
# With square kernels and equal stride
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
m = nn.Conv2d(512, 256, 2, stride=2)
# m = m.flatten(2)
# m = m.transpose(-2,-1)
# params = nn.Parameter(torch.zeros(1,512,256))
# m = m+params

zeros = torch.zeros(1,3,512,256)
input = torch.randn(1,3,512,256)
add = input+zeros
# non-square kernels and unequal stride and with padding
# m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = torch.randn(1,512,28,28)
output = m(input)
print(output.size())
# exact output size can be also specified as an argument
input = torch.randn(1, 16, 12, 12)
Dilated_downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1,dilation=3)
Dilated_upsample =   nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1,dilation = 3)


downsample  = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample    = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1)

d_h = Dilated_downsample(input)
h = downsample(input)
up = upsample(h)
d_up = Dilated_upsample(d_h)
print(h.size())
up = upsample(h)
print(up.size())
torch.Size([1, 16, 6, 6])
output = upsample(h, output_size=input.size())
output.size()
torch.Size([1, 16, 12, 12])
input = img[np.newaxis,:,:,:]
input = torch.tensor(input)
input = torch.moveaxis(input,-1,1)
input = input.float()
in_channels = 64
out_channels = 128

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.double = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.double(x)

model =  nn.Sequential(
                nn.Conv2d(3,in_channels,3,stride=1,dilation=1,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 3, stride=1,dilation=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels,3,kernel_size=3,stride=1,dilation = 1),
                # nn.Conv2d(out_channels,3,1)
)
model(input)
from torchsummary import summary
summary(model,input_size=(3,480,640),batch_size=1,device='cpu')


ones = torch.ones((1,64,160,240))
zeros = torch.zeros((1,64,124,204))

ones[:,:,:zeros.shape[2],:zeros.shape[3]] = zeros
test = ones.numpy()
new = ones[:,:,:zeros.shape[2],:zeros.shape[3]]



import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
def overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
        resize: tuple[int, int] = (640,480)
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.

    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.

    Returns:
        image_combined: The combined image.

    """
    color = np.asarray(color).reshape(3, 1, 1)
    print(color.shape)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    print(colored_mask.shape)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    print(masked.shape)
    image_overlay = masked.filled()
    print(image_overlay.shape)
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    print(image_combined.shape)

    return image_combined


img = cv2.imread(r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/9/original_0.png')
mask = cv2.imread(r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/9/pred_0.png')
img = img[:,:,0]
mask = mask[:,:,0]
# image_combined = overlay(img,mask)
fig,ax = plt.subplots()
ax.imshow(img,cmap='gray')
ax.imshow(mask,cmap='Blues',alpha=.5)
fig.show()

input = r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/10x5/IR/squiggle/Focal+Dice/Input'
preds = r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/10x5/IR/squiggle/Focal+Dice/Pred'

input_list = sorted(os.listdir(input))
pred_list  = sorted(os.listdir(preds))
zip_list = list(zip(input_list,pred_list))
random.shuffle(zip_list)
input_list,pred_list = zip(*zip_list)
input_img = []
pred_img  = []
for i in input_list:
    input_img.append([cv2.cvtColor(cv2.imread(os.path.join(input,i)),cv2.COLOR_BGR2RGB)])
for i in pred_list:
    pred_img.append([cv2.imread(os.path.join(preds, i))])

input_img = np.array(input_img).squeeze()
pred_img = np.array(pred_img).squeeze()

pred_img  = pred_img[:,:,:,0]

#define subplot grid
for i in range(0,len(input_img),4):
    figs,axs = plt.subplots(2,2,figsize=(15,12))
    plt.subplots_adjust(hspace=.2)
    figs.suptitle("10 epoch Train dilation 20x10")
    input_img_current = input_img[i:i+4]
    pred_img_current = pred_img[i:i+4]
    input_list_current = input_list[i:i+4]
    for img,preds,names,ax in zip(input_img_current,pred_img_current,input_list_current,axs.ravel()):
        ax.imshow(img)
        ax.imshow(preds, cmap='Greens', alpha=.5)

        #formatting
        ax.set_title(names)
        figs.savefig(f'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/10x5/IR/squiggle/Focal+Dice/plot/plot{i}.png')
    plt.show()


img = cv2.cvtColor(cv2.imread('/home/ethan/PycharmProjects/UNET_pytorch/boatPhotos1/Image061.jpg'),cv2.COLOR_BGR2RGB)


img = np.moveaxis(img,1,0)
count = np.count_nonzero(img==255)
plt.imshow(img)
plt.show()


##############Resize and Add all preds together#############
image_list = []
for i in range(3):
    run = f'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/N_5CatRun/DiffResEVAL/Pred_{i}'
    image_names = sorted(os.listdir(run))
    for j in image_names:
        image = os.path.join(run,j)
        temp = cv2.resize(cv2.imread(image),(240,160))
        image_list.append([temp])
    if i == 0:
        pred_0 = np.squeeze(np.array(image_list))
    elif i == 1:
        pred_1 = np.squeeze(np.array(image_list))
    else:
        pred_2 = np.squeeze(np.array(image_list))
    image_list = []


arr = pred_0+pred_1+pred_2
arr[arr>0] = 255
input = r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/N_5CatRun/0'
input_img = []
input_list = sorted(os.listdir(input))
for i in input_list:
    input_img.append([cv2.cvtColor(cv2.imread(os.path.join(input,i)),cv2.COLOR_BGR2RGB)])


input_img = np.array(input_img).squeeze()
for i in range(0,len(input_img),4):
    figs,axs = plt.subplots(2,2,figsize=(15,12))
    plt.subplots_adjust(hspace=.2)
    figs.suptitle("8 Mil Params 100Epoch Train IR")
    input_img_current = input_img[i:i+4]
    pred_img_current = arr[i:i+4]
    input_list_current = input_list[i:i+4]
    for img,preds,names,ax in zip(input_img_current,pred_img_current,input_list_current,axs.ravel()):
        ax.imshow(img)
        ax.imshow(preds, cmap='Greens', alpha=.7)

        #formatting
        ax.set_title(names)
        figs.savefig(f'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/N_5CatRun/DiffResEVAL/plots/plot{i}.png')
    plt.show()


########Patching########
# import torch
# h,w = 160,160
# patch_size = 8
# patches = []
# stride = np.int(patch_size)
# img = cv2.resize(cv2.imread(r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/NAWCAD/0/original_1.png'),(h,w))
# img = img[np.newaxis,:,:,:]
# img = np.moveaxis(img,-1,1)
# img = torch.tensor(img)
# for i in range(0, img.shape[2], stride):
#     temp = img[i:i + stride, i:i + stride, :]
#     patches.append([temp])
#
# patches = np.squeeze(np.array(patches))
#
# patches = patches[np.newaxis,:,:,:]
# batch_size = 1
# patch_dims = patches.shape[-1]
# reshape = np.reshape(patches,(batch_size,-1,patch_dims))
# original = np.reshape(reshape)
