import os

import torch
import torchvision
from dataset import BoatsDataset,IRBoats,IRBoatsDuplicates,EVAL,SegDataPoints
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np



validation_split = .2
shuffle_dataset = True
def save_checkpoint(state, filename=f"my_checkpoint_New.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])

def get_loaders(
    train_dir=None,
    train_img_dir=None,
    train_maskdir=None,
    val_dir=None,
    val_img_dir=None,
    val_maskdir=None,

    batch_size=None,
    train_transform=None,
    val_transform=None,
    num_workers=4,
    pin_memory=True,
    eval_dir=None,
    squiggle = None,
    point_annotations = None
):
    if eval_dir is None and squiggle is None:
        #Create duplicates
        train_ds = BoatsDataset(
            dataframe=train_maskdir,
            root_dir=train_dir,
            percentage = 1,
            transform=train_transform,
        )
        # #Create duplicates
        # train_ds = IRBoatsDuplicates(
        #     dataframe=train_maskdir,
        #     root_dir=train_dir,
        #     percentage = .05,
        #     transform=train_transform,
        # )
        #Dont create duplicates
        # train_ds = IRBoats(
        #     image_names=train_img_dir,
        #     dataframe=train_maskdir,
        #     root_dir=train_dir,
        #     percentage = .1,
        #     transform=train_transform,
        # )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,

            shuffle=True
        )
        val_ds = BoatsDataset(
            root_dir=val_dir,
            dataframe=val_maskdir,
            transform=val_transform,
            percentage=1,
        )

        # val_ds = IRBoatsDuplicates(
        #     root_dir=val_dir,
        #     dataframe=val_maskdir,
        #     transform=val_transform,
        #     percentage=1,
        # )
        # val_ds = IRBoats(
        #     image_names=val_img_dir,
        #     root_dir=val_dir,
        #     dataframe=val_maskdir,
        #     transform=val_transform,
        #     percentage=1,
        # )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
        return train_loader, val_loader
    elif squiggle:
        squiggle_ds = SegDataPoints(
            point_annotations=point_annotations,
            transforms =train_transform
        )
        train_loader = DataLoader(
            squiggle_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        return train_loader
    else:
        EVAL_ds = EVAL(
            root_dir=eval_dir,
            transform=val_transform
        )
        EVAL_loader = DataLoader(
            EVAL_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        return EVAL_loader


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer,patience, criterion,save_path,
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, save_path)
            patience = 0
            return patience
        else:
            return patience+1


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')


def check_accuracy(loader, model,device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    tracker = []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # y = torch.moveaxis(y,-1,1)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            # print('y shape: ', y.shape)
            # print('preds shape: ',preds.shape)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, epoch,folder="saved_images/", device="cuda"
):
    model.eval()
    newpath = folder+'/'+str(epoch)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for idx, x in enumerate(loader):
        x = x.to(device=device)

        # y = torch.moveaxis(y,-1,1)
        # print(y.shape[0], y.unsqueeze(1).shape[0])
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            # print(preds.shape)
        size = preds.size()
        length = size[0]
        for i in range(length):
            torchvision.utils.save_image(
                x[i],f'{newpath}/original_{idx}.png',padding=0
            )
        for i in range(length):
            torchvision.utils.save_image(
                preds[i], f"{newpath}/pred_{idx}.png"
            )
        # for i in range(length):
        #
        #     torchvision.utils.save_image(
        #         y[i], f"{newpath}/true_{idx}.png",padding=0
        #     )

    model.train()