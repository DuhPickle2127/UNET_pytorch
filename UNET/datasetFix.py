import pandas as pd
import numpy as np
from PIL import Image
import os

#### SEA BUS #####
df = pd.read_csv(r'/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/train_ship_segmentations_v2.csv')
df["EncodedPixels"] = df["EncodedPixels"].fillna('0')
img_path = r'/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/train_v2'
img_names = sorted(os.listdir(img_path))

test = df.loc[0]
count = 0
file_name = []
encoded_pixels = []
for i in range(len(img_names)):
    test = df.loc[i]
    if test["ImageId"] in img_names:
        file_name.append(test["ImageId"])
        encoded_pixels.append(test["EncodedPixels"])

file_name = np.array(file_name)
encoded_pixels = np.array(encoded_pixels)
expand1 = np.expand_dims(file_name,axis=1)
expand2 = np.expand_dims(encoded_pixels,axis=1)
concat = np.concatenate((expand1,expand2),axis=1)
new = pd.DataFrame({"ImageId":file_name,"EncodedPixels":encoded_pixels})

# new.to_csv("/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/TrainSegNew.csv")
import random
newdf = pd.read_csv(r'/home/ethan/PycharmProjects/UNET_pytorch/ETHAN_DATA/TrainSegNew.csv')

test = newdf.groupby('ImageId')
grouped_lists = test["EncodedPixels"].apply(list)
grouped_lists = grouped_lists.reset_index()
shuffled = np.random.permutation(grouped_lists)
shuffled_df = pd.DataFrame(shuffled,columns = ["ImageId","EncodedPixels"])
tester = shuffled_df["EncodedPixels"][9419]
imagename = shuffled_df["ImageId"][9419]
name = '000194a2d.jpg'
tester = np.reshape(tester,(len(tester),1))
names = np.full(tester.shape,"imagename")
concat = np.concatenate((names,tester),axis=1)

tester1 = shuffled_df["EncodedPixels"][9500]
imagename1 = shuffled_df["ImageId"][9500]
name = '000194a2d.jpg'
tester1 = np.reshape(tester1,(len(tester1),1))
names1 = np.full(tester1.shape,"imagename")
concat1 = np.concatenate((names1,tester1),axis=1)

stack = np.concatenate((concat,concat1),axis=0)
tester1 = np.empty((len(tester[0]),len(tester[0])),dtype='str')


train = np.empty((1,2))

val = np.empty((1,2))

percentage = .8
for i in range(len(shuffled_df)):
    print(i)
    imagename = shuffled_df["ImageId"][i]
    pixel_vals = shuffled_df["EncodedPixels"][i]
    pixel_vals = np.reshape(pixel_vals,(len(pixel_vals),1))
    # print(pixel_vals.shape)
    names = np.full(pixel_vals.shape,imagename)
    # print(names.shape)
    concat = np.concatenate((names,pixel_vals),axis=1)
    # print(concat)
    if (i/len(grouped_lists)) <= percentage:
        train = np.concatenate((train,concat),axis=0)
    else:
        val = np.concatenate((val,concat),axis=0)

newtrain = train[1:]
newval = val[1:]
train_csv = pd.DataFrame(newtrain,columns = ["ImageId","EncodedPixels"])
val_csv   = pd.DataFrame(newval,columns = ["ImageId","EncodedPixels"])

train_csv.to_csv("train_csv")
val_csv.to_csv("val_csv")

##### IR NAWCAD #####

import pandas as pd
import numpy as np
from PIL import Image
import os

import random

# groups = [df for _, df in df.groupby('sampleID')]
# random.shuffle(groups)
#
# pd.concat(groups).reset_index(drop=True)
csv1 = pd.read_csv(r'/home/ethan/PycharmProjects/UNET_pytorch/boatPhotos2.csv').to_numpy()
csv2 = pd.read_csv(r'/home/ethan/PycharmProjects/UNET_pytorch/boatPhotos4.csv').to_numpy()

merged = np.vstack((csv1,csv2))

merged_data = pd.DataFrame(merged,columns=['Image_Name','Bx','By','Cx','Cy'])
groups = [merged_data for _, merged_data in merged_data.groupby('Image_Name')]
random.shuffle(groups)

pd.concat(groups).reset_index(drop=True)

arr = np.array(groups)
reshape = np.reshape(arr,(arr.shape[0]*5,5))
data = pd.DataFrame(reshape,columns=['Image_Name','Bx','By','Cx','Cy'])


eighty = data[:int(len(data)*.8)]
twenty = data[int(len(data)*.8):]

train_images = eighty.groupby("Image_Name").agg(list)
val_images = twenty.groupby("Image_Name").agg(list)

train_images = pd.DataFrame(np.array(train_images.index).reshape(-1,1),columns=["Image_Name"])
val_images = pd.DataFrame(np.array(val_images.index).reshape(-1,1),columns=["Image_Name"])

eighty.to_csv("IR_FULL_train_annots")
twenty.to_csv("IR_FULL_val_annots")
train_images.to_csv("IR_train1_csv_Images")
val_images.to_csv("IR_val1_csv_Images")

test = merged_data.groupby('Image_Name').agg(list)
grouped_lists = test['Bx','By','Cx','Cy']
# test["Image_Name"] = merged[:,0]
# grouped_lists = grouped_lists.reset_index()
shuffled = np.random.permutation(test)

shuffled_df = pd.DataFrame(shuffled,columns = ['Bx','By','Cx','Cy'])
shuffled_df["ImageName"] = test.index
#
#
# imagename = shuffled_df["ImageId"][0]
# for i in imagename:
#     print(merged_data.loc[merged_data["Image_Name"==i]])

train = np.empty((1,5))
val = np.empty((1,5))

percentage = .8
for i in range(len(shuffled_df)):
    print(i)
    imagename = test.index[i]
    print(imagename)
    Bx = np.array(shuffled_df.iloc[i][0]).reshape(-1,1)
    By = np.array(shuffled_df.iloc[i][1]).reshape(-1,1)
    Cx = np.array(shuffled_df.iloc[i][2]).reshape(-1,1)
    Cy = np.array(shuffled_df.iloc[i][3]).reshape(-1,1)
    # print(Bx,By,Cx,Cy)
    pixels = np.hstack((Bx,By,Cx,Cy))
    pixels_shape = pixels.shape

    names = np.full((pixels.shape[0],1),imagename)
    # print(names.shape)
    concat = np.concatenate((names,pixels),axis=-1)
    # print(concat)
    if (i/len(grouped_lists)) <= percentage:
        train = np.concatenate((train,concat),axis=0)
    else:
        val = np.concatenate((val,concat),axis=0)

newtrain = train[1:]
newval = val[1:]
IR_train_csv = pd.DataFrame(newtrain,columns=['Image_Name','Bx','By','Cx','Cy'])
IR_val_csv   = pd.DataFrame(newval,columns=['Image_Name','Bx','By','Cx','Cy'])
IR_train_csv.to_csv("IR_train1_csv")
IR_val_csv.to_csv("IR_val1_csv")
test_arr = np.zeros((5,1))
test_ones = np.ones((5,4))

eighty = shuffled_df[:int(len(shuffled_df)*.8)]
twenty = shuffled_df[int(len(shuffled_df)*.8):]
eighty.to_csv("IR_train_csv")
twenty.to_csv("IR_val_csv")

shuffled_df.to_csv("shuffled",index=False,float_format='float64')
