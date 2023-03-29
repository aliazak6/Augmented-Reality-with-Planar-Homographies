import cv2
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches

sigma = 0.15

def matchPics(I11, I22):
	"""
	Matches features between two images given a set of matching points.
	Inputs:
	I1, I2 - Images to be matched
	Outputs:
	matches - Mx2 matrix specifying the indices of matched features in I1 and I2 respectively
	locs1 - Feature locations in I1
	locs2 - Feature locations in I2
	"""
	
	#Convert Images to GrayScale
	I1 = cv2.cvtColor(I11, cv2.COLOR_BGR2GRAY)
	I2 = cv2.cvtColor(I22, cv2.COLOR_BGR2GRAY)
	#Detect Features in Both Images
	locs1 = corner_detection(I1,sigma)
	locs2 = corner_detection(I2,sigma)
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1, locs1)
	desc2, locs2 = computeBrief(I2, locs2)
	#Match features using the descriptors
	matches = briefMatch(desc1,desc2)
	#plotMatches(I11,I22,matches,locs1,locs2)
	return matches, locs1, locs2

if __name__ == '__main__':
	import os
	dirname = os.path.dirname(__file__)
	cover_name = os.path.join(dirname, '../data/cv_cover.jpg')
	desk_name = os.path.join(dirname, '../data/cv_desk.png')
	#Read images
	cover_img = cv2.imread(cover_name)
	desk_img = cv2.imread(desk_name)
	#Compute homography
	matches, locs1, locs2 = matchPics(cover_img, desk_img)
	plotMatches(cover_img,desk_img,matches,locs1,locs2)
