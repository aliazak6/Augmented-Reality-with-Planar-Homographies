import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from matchPics import matchPics
import planarH
import os
#Read images
dirname = os.path.dirname(__file__)
cover_name = os.path.join(dirname, '../data/cv_cover.jpg')
desk_name = os.path.join(dirname, '../data/cv_desk.png')
hp_cover_name = os.path.join(dirname, '../data/hp_cover.jpg')
cover_img = cv2.imread(cover_name)
desk_img = cv2.imread(desk_name)
hp_cover_img = cv2.imread(hp_cover_name)

# Resize hp_cover to match cover_img
h, w = cover_img.shape[0], cover_img.shape[1]
resized = np.zeros(cover_img.shape)
resized[..., 0] = cv2.resize(hp_cover_img[..., 0], (w, h))
resized[..., 1] = cv2.resize(hp_cover_img[..., 1], (w, h))
resized[..., 2] = cv2.resize(hp_cover_img[..., 2], (w, h))
hp_cover_img = resized.astype(np.uint8)

#Compute homography
matches, locs1, locs2 = matchPics(cover_img, desk_img)
matched_locs1 = locs1[matches[:,0],:]
matched_locs2 = locs2[matches[:,1],:]
homography, inliers = planarH.computeH_ransac(matched_locs1, matched_locs2)
# Warp images
composite_img = planarH.compositeH( homography, desk_img, hp_cover_img)
cv2.imwrite('composite_img.jpg', composite_img)
cv2.imshow('composite_img', composite_img)
cv2.waitKey(0)
