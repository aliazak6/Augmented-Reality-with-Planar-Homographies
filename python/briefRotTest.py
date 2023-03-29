import numpy as np
import cv2
from matchPics import matchPics
from scipy import ndimage
import matplotlib.pyplot as plt
import os
#Q3.5

dirname = os.path.dirname(__file__)
cover_name = os.path.join(dirname, '../data/cv_cover.jpg')
img = cv2.imread(cover_name)
hist = np.zeros(36)
for i in range(36):
	#Rotate Image
	rotated = ndimage.rotate(img, i*10, reshape=False)
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, rotated)
	#Update histogram
	hist[i] = len(matches)

#Display histogram
plt.plot(hist)
plt.show()


