
# ------------------ CONFIG ------------------

# Imports
import cv2
from glob import glob
import os
import xml.etree.ElementTree as ET
from random import randint
import random
import json
import sys

# Image, label, and mask directories
image_dir = "/home/ethan/PycharmProjects/UNET_pytorch/IR/"
mask_dir = "../YOLO-V5/data/IR/masks"
FORMAT="jpg"

# Choose which boat photos folder is used
boatPhotosDir = sys.argv[1]
image_dir += "/" + boatPhotosDir

# Create mask directory if doesn't exist already (kaggle dataset doesn't by default)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
mask_dir += "/" + boatPhotosDir
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# List of image files
image_files = glob(image_dir + "/*."+FORMAT)
print(image_files)



# If overwrite is true, start from the beginning and overwrite existing mask annotations
# If false, start from next non-annotated image
OVERWRITE = False
# Do not write at all if testing is true
TESTING = False
# Number of random points to be annotated per image
NPOINTS = 32


# List of points per image
points = {}

drawing = False
polarity = -1

# ------------------ HELPERS ------------------

# Function to add point to points data structure
def add_point(basename, x, y, polarity):
    coords = set([tuple(p[:2]) for p in points[basename]])

    if not (x, y) in coords:
        points[basename].append([x, y, polarity])
    else:
        print("Dupe!")

# Mouse click handler
def mouse_click(event, x, y, flags, param):
    global drawing, polarity
    # Left button down -> part of the ship mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        polarity = 1
        #print("Drawing - ship")
    # Right button down -> part of the background mask
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing=True
        polarity = -1
        #print("Drawing - background")
    # If mouse moves and drawing is enabled, mark points
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            #print("Moving + adding!")
            add_point(basename, x, y, polarity)
    # If mouse button is back up, stop drawing
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        drawing=False
        #print("Stopped drawing")



# Bind mouse click handler to window
cv2.namedWindow("window",cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback("window", mouse_click)

# Function to scale up an image to 640p resolution
def upscale(img, w=640):
    h = int(img.shape[1] / img.shape[0] * w)
    return cv2.resize(img, dsize=(h, w))


# ------------------ PROCESSING ------------------

# Iterate through image files...
processed_num = 0
for imf in image_files:
    
    # Print annotation progress
    processed_num += 1
    print(f"Processing {processed_num} / {len(image_files)}")

    # Read basename
    basefile = os.path.basename(imf)
    basename = os.path.splitext(basefile)[0]

    # Add a points entry for this file if doesn't exist yet
    if not basename in points:
        points[basename] = []

    # Get path to corresponding label and mask
    maskf = mask_dir + "/" + basename + ".json"

    # Ignore if overwriting is false and a file already exists
    if (not OVERWRITE and os.path.exists(maskf)):
        print(maskf)
        print(f"OVERWRITE IS FALSE: Point sample .json already exists for image {basename}")
        continue
        


    # Read image
    im = cv2.imread(imf)

    # Keep looping until a new point has been clicked
    while True:
    
        # Create a new image
        newim = im.copy() 
        print(newim.shape)
        newim = upscale(newim)
        print(newim.shape)

        # Get current number of points
        pid = len(points[basename])
        
        # Draw current detections
        counts = {-1: 0, 1: 0}

        for sample in points[basename]:
            (x, y, polarity) = tuple(sample)
            color = (0, 255, 0) if polarity == 1 else (0, 0, 255)
            counts[polarity] += 1

            cv2.circle(newim, (x, y), 2, color, -1)

        # Draw instructional text, display how many points left to click
        cv2.putText(newim, f"Points drawn: {pid}/{NPOINTS}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        cv2.putText(newim, f"Background points selected: {counts[-1]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        cv2.putText(newim, f"Ship points selected: {counts[1]}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Draw image
        cv2.imshow("window", newim)

        # Stall until a point has been clicked + identified
        incomplete = False
        complete = False
        while True:
            k = cv2.waitKey(33)

            # Once the number of points has increased, move on
            if len(points[basename]) > pid:
                #print("Increased!")
                break

            # If the "Q" button has been clicked, invalidate current annotations and raise a flag
            if k == 113: 
                print("Quitting and invalidating current image's annotations.")
                incomplete = True
                exit()
            
            # If the "r" button has been clicked, reset the detections and remove all clicked points
            if k == 114:
                print("Resetting current image's annotations!")
                points[basename] = []
                break
            
            # If the spacebar has been clicked, AND we have more than N points, sample N points from each class
            if k == 32:
                # If enough points have been selected...
                if len(points[basename]) >= NPOINTS:
                    # Raise the completion flag
                    print(f"{len(points[basename])} points selected! Good to go.")


                    # Grab the background points and ship points
                    background_points = [p for p in points[basename] if p[2] == -1]
                    ship_points = [p for p in points[basename] if p[2] == 1]
                    
                    # If not enough ship points, double check to make sure that is ok:
                    inp = "none"
                    if len(ship_points) < 8:
                        print("Not enough ship points! Need at least 8.")
                        inp = input("Continue? (y/n)")
                        if inp == "y":
                            complete = True
                        else:
                            break
                    else:
                        complete = True

                    # Calculate how many of each point set to sample proportionally
                    nship = round(NPOINTS * len(ship_points) / (len(background_points) + len(ship_points)))
                    # Get at least 8 points, more if proportional
                    if inp == "none":
                        nship = max(nship, 8)

                    nbackground = NPOINTS - nship

                    # Sample new set of points
                    points[basename] = random.sample(background_points, nbackground) + random.sample(ship_points, nship)
                    # break
                    break
                else:
                    print("Not quite enough points! Keep going.")

        # If "incomplete", delete current entry and break out of loop
        if incomplete:
            del points[basename]
            break
        
        # If "complete", break out of the loop - you are done!
        if complete:
            break
    
    print(f"Image {basename} processed!")
    #print(points[basename])

    # Prepare annotation json output
    outdata = {"samples": points[basename]}
    print(outdata)
    outdata_serialized = json.dumps(outdata, indent=4)

    # Write to file if applicable
    if not TESTING:
        outfile = open(maskf, "w+")
        outfile.write(outdata_serialized)
        outfile.close()