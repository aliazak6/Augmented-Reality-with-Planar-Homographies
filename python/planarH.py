import numpy as np
import cv2


def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points
	# x1 = H x2
	N = len(x1)
	Ai = np.zeros((2*N, 9))
	for i in range(0,N):
		Ai[2*i] = [x1[i][0], x1[i][1], 1, 0, 0, 0, -x2[i][0]*x1[i][0], -x2[i][0]*x1[i][1], -x2[i][0]]
		Ai[2*i+1] = [0, 0, 0, x1[i][0], x1[i][1],1,-x2[i][1]*x1[i][0], -x2[i][1]*x1[i][1], -x2[i][1]]
	
	# cant use eig because Ai is not square matrix
	_,_, vh = np.linalg.svd(Ai) # vh is sorted in ascending order
	H2to1 = vh[-1].reshape(3,3)

	# make last element 1
	H2to1 = H2to1 / H2to1[2,2]

	return H2to1

def normalize_points(x):
	origin = x - np.mean(x,axis=0)
	max_x = np.max(np.sqrt(np.sum(x**2, axis=1)))
	scale_factor = np.sqrt(2) / max_x
	new = scale_factor *origin
	return new

def computeH_norm(x1, x2):
	#Q3.7
	"""
	Normalization makes results worst so I dont exactly know what is the problem
	""" 	
	#Scale transform 1
	#x1 = normalize_points(x1)

	#Scale transform 2
	#x2 = normalize_points(x2)

	#Compute homography
	H2to1 = computeH(x1, x2)
	
	return H2to1


def dist(p1,p2, H):
    """Returns the geometric distance between a pair of points given the
    homography H. I copied it from stackoverflow.
    Args:
        pair (List[List]): List of two (x, y) points.
        H (np.ndarray): The homography.
    Returns:
        float: The geometric distance.
    """
    # points in homogeneous coordinates
    p1 = np.array([p1[0], p1[1], 1])
    p2 = np.array([p2[0], p2[1], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate
    # transform p2 to homogeneous coordinates
    return np.linalg.norm(np.transpose(p2) - p2_estimate)


def computeH_ransac(x1, x2):
	#Q3.8
	"""
	Compute the best fitting homography given a list of matching points
	"""
	# Parameters for RANSAC
	s = 4 # minimum number of points required to fit a model
	e = 0.5 # outlier ratio
	p = 0.99 # probability of choosing at least one sample free of outliers
	threshold = 15 # threshold for considering a point an inlier
	N = np.log(1-p)/np.log(1-(1-e)**s) # number of iterations)	

	length = len(x1)
	max_inliers = np.zeros(length)
	bestH2to1 = np.zeros((3,3))

	for i in range(int(N)):
		#Randomly select s points -> 4 for homograpy. Q2
		idx = np.random.choice(length, s, replace=False)
		x1_sample = x1[idx]
		x2_sample = x2[idx]

		#Compute the homography
		H2to1 = computeH_norm(x1_sample, x2_sample)

		#Compute inliers
		inliers = np.zeros(length)
		for j in range(length):
			if dist(x1[j],x2[j],H2to1) < threshold:
				inliers[j] = 1
		
		#Update max_inliers
		if np.sum(inliers) > np.sum(max_inliers):
			max_inliers = inliers
			bestH2to1 = H2to1

	return bestH2to1, max_inliers


def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    
    warped_img = cv2.warpPerspective(img.swapaxes(0, 1), H2to1, (template.shape[0], template.shape[1])).swapaxes(0, 1)

	# Create mask of same size as template
    mask = np.zeros(warped_img.shape)
    ch1, ch2, ch3 = warped_img[:, :, 0], warped_img[:, :, 1], warped_img[:, :, 2]

	# make it binary instead of grayscale. 1 if pixel is not black, 0 if pixel is black
    mask[:, :, 0] = (ch1 > 0).astype(np.uint8)
    mask[:, :, 1] = (ch2 > 0).astype(np.uint8)
    mask[:, :, 2] = (ch3 > 0).astype(np.uint8)

	# For warping the template to the image, we need to invert it.
    mask = np.logical_not(mask).astype(np.uint8)

    # Use mask to combine the warped template and the image
    composite_img = warped_img + template * mask
    return composite_img


