import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

###AIR BUS KAGGLE###
# dir         = r'ETHAN_DATA'
# folder_name = r'train_v2'
# csv = pd.read_csv(r'/home/ethan/PycharmProjects/UNET_pytorch/train_csv')
# image_names = csv["ImageId"]
# folder      = os.path.join(dir,folder_name)


#### NAWCAD ######
dir         = r'IR'
folder_name = r'boatPhotos2'
folder      = os.path.join(dir,folder_name)
image_names = os.listdir(folder)

annots      = []
NAME        = []
BACKGROUND  = []
CLASS       = []

def mousePoints(event, x, y, flags, params):
    global counter
    global LCOUNT,RCOUNT

    # Left button mouse click event opencv
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     Class[LCOUNT] = x,y
    #     cv2.circle(img, (Class[LCOUNT][0], Class[LCOUNT][1]), 2, (0, 255, 0), cv2.FILLED)
    #     LCOUNT += 1


    ####FOR 5 CLICK SAMPLES####
    if event == cv2.EVENT_LBUTTONDOWN:
        Background[LCOUNT] = x,y
        cv2.circle(img, (Background[LCOUNT][0], Background[LCOUNT][1]), 3, (0, 255, 0), cv2.FILLED)
        LCOUNT += 1
    if event == cv2.EVENT_MBUTTONDOWN:
        Class[RCOUNT] = x,y
        cv2.circle(img, (Class[RCOUNT][0], Class[RCOUNT][1]), 3, (255, 0, 0), cv2.FILLED)
        RCOUNT += 1

for i in range(1600):
    image       = os.path.join(folder,image_names[i])
    img         = cv2.imread(image)
    name        = np.empty((5,1)).astype('str')
    name[:,0]   = image_names[i]
    Background  = np.zeros((5, 2),np.int)
    Class       = np.zeros((5, 2),np.int)
    counter     = 0
    LCOUNT      = 0
    RCOUNT      = 0
    while RCOUNT < 6:
        cv2.putText(img,text='Lbutton=Class',org=(15,15),fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=.5, color=(0, 255, 0),thickness=1)
        cv2.putText(img,text='Rbutton=Background',org=(150,15),fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=.5, color=(255, 0,0),thickness=1)
        cv2.putText(img, text=f'{str(len(image_names)-i)} images left', org=(400,15), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=.25, color=(0,0,255), thickness=1)
        cv2.imshow("Original Image ", img)
        # Mouse click event on original image
        cv2.setMouseCallback("Original Image ", mousePoints)
        cv2.waitKey(1)
        if cv2.waitKey(20) == 27:
            break
    # while RCOUNT <5 or LCOUNT <5:
    #     # Showing original image
    #     cv2.putText(img,text='Lbutton=Background,Mbutton=Class',org=(15,15),fontFace=cv2.FONT_HERSHEY_TRIPLEX,
    #                 fontScale=.5, color=(0, 255, 0),thickness=1)
    #     cv2.putText(img, text=f'{str(len(image_names)-i)} images left', org=(400,15), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
    #                 fontScale=.25, color=(0,0,255), thickness=1)
    #     cv2.imshow("Original Image ", img)
    #     # Mouse click event on original image
    #     cv2.setMouseCallback("Original Image ", mousePoints)
    #     cv2.waitKey(1)
    #     if cv2.waitKey(20) == 27:
    #         break
    #

    cv2.destroyAllWindows()
    NAME.append([name])
    BACKGROUND.append([Background])
    CLASS.append([Class])


NAME        = np.squeeze(np.array(NAME).astype('str'))
# BACKGROUND  = np.squeeze(np.array(BACKGROUND))
CLASS       = np.squeeze(np.array(CLASS))
NAME        = NAME.reshape(-1,1)
# BACKGROUND  = np.reshape(BACKGROUND,(BACKGROUND.shape[0]*BACKGROUND.shape[1],2))
# test       = np.reshape(CLASS,(CLASS.shape[0]*CLASS.shape[1],2))
CLASS       = np.reshape(CLASS,(CLASS.shape[0]*CLASS.shape[1],2))
data        = pd.DataFrame(NAME,columns=['Image_Name'])
# data["Bx"]  = BACKGROUND[:,:1]
# data['By']  = BACKGROUND[:,1:]
data['Cx']  = CLASS[:,:1]
data['Cy']  = CLASS[:,1:]

data.to_csv(f'{folder_name}.csv',index=False)




class DrawLineWidget(object):
    def __init__(self,img):
        self.original_image = cv2.imread(img)
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone
    def image_coords(self):
        return self.image_coordinates


draw_line_widget = DrawLineWidget(r'/home/ethan/PycharmProjects/UNET_pytorch/IR/boatPhotos2/Image4209.jpg')
while True:

    cv2.imshow('image', draw_line_widget.show_image())
    key = cv2.waitKey(1)
    image_coords = draw_line_widget.image_coords()
    # Close program with keyboard 'q'
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(1)

import matplotlib.pyplot as plt
arr = np.uint(np.zeros((draw_line_widget.show_image().shape)))

arr[214:217,286:309] = 1
img = cv2.imread(r'/home/ethan/PycharmProjects/UNET_pytorch/IR/boatPhotos2/Image4209.jpg')
img = img[:,:,0]
arr = arr[:,:,0]
plt.imshow(img,cmap ='gray')
plt.imshow(arr,cmap='Blues',alpha=.5)
plt.show()
x_pixels = np.arange(image_coords[1][1],image_coords[0][1])
y_pixels = np.arange(image_coords[1][1],image_coords[0][1])