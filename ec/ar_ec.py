import numpy as np
import cv2
#Import necessary functions
import sys
sys.path.append('D:\\ubuntu_yedek\\compVision\\assgn2\\python')
from matchPics import matchPics
import planarH
from loadVid import loadVid
import os

#Read images
dirname = os.path.dirname(__file__)
book_path = os.path.join(dirname, '../data/book.mov')
ar_path = os.path.join(dirname, '../data/ar_source.mov')
book_frames = loadVid(book_path)
ar_raw = loadVid(ar_path)
cover_path = os.path.join(dirname, '../data/cv_cover.jpg')
cover_img = cv2.imread(cover_path)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (book_frames[0].shape[1], book_frames[0].shape[0]))

# Resize hp_cover to match cover_img
h, w = cover_img.shape[0], cover_img.shape[1]
resized = np.zeros(cover_img.shape)


ar_frames = np.zeros((len(ar_raw),505,350,3), dtype=np.uint8)

#Resize ar image
for i , ar_img in enumerate(ar_raw):
    w_ar = ar_img.shape[0]
    h_ar = ar_img.shape[1]
    # strip black lines
    ar_img = ar_img[130:h_ar-130,:]
    # adjust height
    ar_img = cv2.resize(ar_img, (w_ar, h+65))
    # crop width
    ar_frames[i] = ar_img[:,w_ar//2-w//2:w_ar//2+w//2]

for i,book_frame in enumerate(book_frames):
    #Compute homography
    matches, locs1, locs2 = matchPics(cover_img, book_frame)
    matched_locs1 = locs1[matches[:,0],:]
    matched_locs2 = locs2[matches[:,1],:]
    homography, inliers = planarH.computeH_ransac(matched_locs1, matched_locs2)

    # Warp images
    if(i<len(ar_frames)):
        composite_img = planarH.compositeH( homography, book_frame, ar_frames[i])
        out.write(composite_img)
    else:
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()
