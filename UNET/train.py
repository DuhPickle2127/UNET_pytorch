import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNET import UNET
import pandas as pd
import torch.nn.functional as F
from NETutils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    SaveBestModel,
    save_model
)
import json
import glob
import os

EXTENSION = "jpg"
imlist = glob.glob(f"IR/*/*.{EXTENSION}")
filenamedict = {os.path.basename(f):f for f in imlist}
point_annotations = glob.glob("IR/masks/*/*.json")

from torchsummary import summary
# from pytorch_model_summary import summary
#Hyperparams
LEARNING_RATE = 1e-3
DEVICE        = 'cuda' if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 16
NUM_EPOCHS    = 50
NUM_WORKERS   = 2
IMAGE_HEIGHT  = 160
IMAGE_WIDTH   = 240
NUM_EVALS     = 3
PIN_MEMORY    = True
LOAD_MODEL    = False
TRAIN         = True
SAVE_FOLDER   = r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/10x5/AIRBUS/34Mil/BCEandDice'
EVAL_IMG_DIR  = r'/home/ethan/PycharmProjects/UNET_pytorch/testimgs'
IMG_DIR       = r'/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/train_v2'
# IR_TRAIN_MASK_DIR = r'/home/ethan/PycharmProjects/UNET_pytorch/IR_train1_csv_annots'
# IR_TRAIN_IMG_DIR = r'/home/ethan/PycharmProjects/UNET_pytorch/IR_train1_csv_Images'
# IR_VAL_MASK_DIR = r'/home/ethan/PycharmProjects/UNET_pytorch/IR_val1_csv_annots'
# IR_VAL_IMG_DIR = r'/home/ethan/PycharmProjects/UNET_pytorch/IR_val1_csv_Images'
SAVE_PATH       = r'/home/ethan/PycharmProjects/UNET_pytorch/saved_images/Full_UNET/10x5/AIRBUS/34Mil/BCEandDice/BCE_DICE_TRANSFER_MODEL.pth'
TEST_PATH       = SAVE_PATH
TRAIN_IMG_DIR   = r'/home/ethan/PycharmProjects/UNET_pytorch/train_csv'
VAL_IMG_DIR     = r'/home/ethan/PycharmProjects/UNET_pytorch/val_csv'
# TRAIN_MASK_DIR['EncodedPixels'] = TRAIN_MASK_DIR['EncodedPixels'].fillna('0')
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1,alpha=.25,gamma=2):
        # comment out if your MTUmodel contains a sigmoid or equivalent activation layer
        shapes = inputs.shape
        # print(inputs.shape)
        inputs = torch.sigmoid(inputs)
        # print(targets.shape)
        # print(inputs.shape)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)

        targets = targets.view(-1)
        weight_a = (alpha*(1-inputs)**gamma*targets)
        weight_b = ((1-alpha)*inputs**gamma*(1-targets))

        focal = torch.mean(torch.log1p(torch.exp(-torch.abs(inputs))+torch.relu(-inputs)) * (weight_a+weight_b) + inputs * weight_b)
        # print(focal.shape)

        intersection = (inputs * targets).sum()


        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        # Dice_FOCAL_BCE = dice_loss + focal

        return dice_loss

class PixelCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PixelCrossEntropyLoss, self).__init__()
        # self.cel = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets, smooth=1,alpha=.25,gamma=2):
        # Flatten inputs, targets, eval mask
        # targets = targets.squeeze(1)
        # print(targets.shape)
        # print(inputs.shape)
        shapes = inputs.shape
        f_inputs = torch.sigmoid(inputs)
        weight_a = (alpha*(1-f_inputs)**gamma*targets)
        weight_b = ((1-alpha)*f_inputs**gamma*(1-targets))
        focal = torch.log1p(torch.exp(-torch.abs(f_inputs))+torch.relu(-f_inputs)) * (weight_a+weight_b) + f_inputs * weight_b
        print(focal.shape)
        p_inputs = focal.permute((0, 2, 3,1)).flatten(end_dim=2)
        p_targets = targets.permute((0, 2, 3,1)).flatten(end_dim=2)
        mask = p_targets
        # evalmask = evalmask.flatten(end_dim=2)
        # print("Input shapes: ", inputs.shape, targets.shape, evalmask.shape, evalmask.min(), evalmask.max())

        # Get indices of points to evaluate at
        indices = torch.nonzero(p_targets)
        # print("Indices have ", indices.shape)
        # Index input and target one-hot classifications, remove extra dim
        p_inputs, p_targets = p_inputs[indices].squeeze(dim=1), p_targets[indices].squeeze(dim=1)
        # print("Squeezed shapes: ", inputs.shape, targets.shape)
        BCE = F.binary_cross_entropy_with_logits(p_inputs, p_targets, reduction='mean')
        print(BCE.shape)
        return BCE

def train_fn(loader,model,optimizer,loss_fn,scaler):
    loop = tqdm(loader)
    # summary(model, (3, IMAGE_HEIGHT, IMAGE_WIDTH))
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data,targets) in enumerate(loop):
        counter += 1
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # L = nn.CrossEntropyLoss()
        #forward float16 training
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            # loss2 = L(loss1,targets)

        train_running_loss += loss.item()
        train_running_correct += (predictions==targets).sum().item()
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm
        loop.set_postfix(loss=loss.item())
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(loader.dataset))
    return epoch_loss, epoch_acc





def main():

    model = UNET(in_channels=3,out_channels=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    # loss_fn = PixelCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-6)
    save_best_model = SaveBestModel()
    if TRAIN:
        IMAGE_HEIGHT = 160
        IMAGE_WIDTH = 160

        train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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
                A.GaussianBlur(),
                ToTensorV2(),
            ],
        )
        val_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.ToGray(p=1.0),
                A.InvertImg(p=.5),
                A.GaussianBlur(),
                ToTensorV2(),
            ],
        )

        train_loader,val_loader = get_loaders(
            train_dir=IMG_DIR,
            # train_img_dir=TRAIN_IMG_DIR,
            train_maskdir=TRAIN_IMG_DIR,
            # val_img_dir=VAL_IMG_DIR,
            val_dir=IMG_DIR,
            val_maskdir=VAL_IMG_DIR,
            batch_size=BATCH_SIZE,
            train_transform=train_transform,
            val_transform=val_transform,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,

        )
  #       train_loader = get_loaders(
  #           squiggle=True,
  #           point_annotations=point_annotations,
  #           num_workers=NUM_WORKERS,
  #           pin_memory=PIN_MEMORY,
  #           batch_size=BATCH_SIZE,
  #           train_transform=train_transform,
  #
  # )
        scaler = torch.cuda.amp.GradScaler()
        load_checkpoint(torch.load(TEST_PATH), model)
        # # # # # # # check_accuracy(val_loader, model, device=DEVICE)
        # save_predictions_as_imgs(val_loader, model, 0,
        #                          folder=SAVE_FOLDER,
        #                          device=DEVICE)
        # print('finished')
        train_loss, train_acc = [], []
        if TRAIN:
            patience = 0
            count = 0

            for epoch in range(NUM_EPOCHS):

                train_epoch_loss, train_epoch_acc = train_fn(train_loader, model, optimizer, loss_fn, scaler)
                train_loss.append(train_epoch_loss)
                train_acc.append(train_epoch_acc)
                patience = save_best_model(train_epoch_loss, epoch, model, optimizer, patience=patience, criterion=loss_fn, save_path=SAVE_PATH)
                count +=1
                if epoch == NUM_EPOCHS - 1:
                    load_checkpoint(torch.load(TEST_PATH), model)
                    # check acc
                    check_accuracy(val_loader, model, device=DEVICE)
                    # save imgs
                    save_predictions_as_imgs(val_loader, model, epoch,
                                             folder=SAVE_FOLDER,
                                             device=DEVICE)
                    print('Train Loss',train_loss)

            # load_checkpoint(torch.load(TEST_PATH), model)
            # # check acc
            # check_accuracy(val_loader, model, device=DEVICE)
            # # save imgs
            # save_predictions_as_imgs(val_loader, model, epoch,
            #                          folder=SAVE_FOLDER,
            #                          device=DEVICE)
            # print('Train Loss', train_loss)

    else:
        IMAGE_HEIGHT = 160
        IMAGE_WIDTH = 240
        eval_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ],
        )
        eval_loader = get_loaders(
            eval_dir=EVAL_IMG_DIR,
            batch_size=1,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            val_transform=eval_transform,
        )

        load_checkpoint(torch.load(TEST_PATH), model)
        # # # # check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(eval_loader, model, 0,
                                 folder=SAVE_FOLDER,
                                 device=DEVICE)
        print('finished')
            # IMAGE_HEIGHT = IMAGE_HEIGHT/2
            # IMAGE_WIDTH  = IMAGE_WIDTH/2


    # check_accuracy(val_loader, MTUmodel, device=DEVICE)
    # scaler = torch.cuda.amp.GradScaler()
    # load_checkpoint(torch.load("/home/ethan/PycharmProjects/UNET_pytorch/my_checkpoint.pth.tar"), model)
    # # check acc
    # load_checkpoint(torch.load(SAVE_PATH), model)
    # # # # check_accuracy(val_loader, model, device=DEVICE)
    # save_predictions_as_imgs(val_loader, model, 9, folder="saved_images/NAWCAD", device=DEVICE)
    # print('finished')
    # train_loss,train_acc = [],[]
    # if TRAIN:
    #     for epoch in range(NUM_EPOCHS):
    #
    #         train_epoch_loss , train_epoch_acc = train_fn(train_loader, model, optimizer, loss_fn, scaler)
    #         train_loss.append(train_epoch_loss)
    #         train_acc.append(train_epoch_acc)
    #         save_best_model(train_epoch_loss,epoch,model,optimizer,criterion=loss_fn,save_path=SAVE_PATH)
    #         # #save MTUmodel
    #         # checkpoint = {
    #         #     "state_dict": model.state_dict(),
    #         #     "optimizer": optimizer.state_dict(),
    #         # }
    #         # save_checkpoint(checkpoint)
    #         # if epoch%10 ==0 and epoch != 0 or epoch == NUM_EPOCHS-1:
    #         if epoch == NUM_EPOCHS - 1:
    #             load_checkpoint(torch.load(SAVE_PATH),model)
    #             #check acc
    #             check_accuracy(val_loader,model,device=DEVICE)
    #             #save imgs
    #             save_predictions_as_imgs(val_loader,model,epoch,folder="/home/ethan/PycharmProjects/UNET_pytorch/saved_images/N_5CatRun",device=DEVICE)

if __name__ == "__main__":
    main()

