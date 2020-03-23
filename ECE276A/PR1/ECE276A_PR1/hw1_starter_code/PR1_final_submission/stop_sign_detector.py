'''
ECE276A WI20 HW1
Stop Sign Detector
Joseph Bell
2/2/2020
'''

import os, cv2
from skimage.measure import label, regionprops
import math
import numpy as np
import matplotlib.pyplot as plt
import copy


class StopSignDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		''' 
		#initializing likelihood model
		self.iterations = 100
		#self.omega = np.array([[0, 0, 0]])
		self.omega = np.array([[-2.07225554], [-8.15944997],  [3.54137508]]) #100 iterations learned parameters
		self.delta_omega = []
		self.likelihood_model = 0
		self.alpha = 0.1
		self.gradient = np.array([[0, 0, 0]])
		self.data_matrix = np.array([[0,0,0]])
		self.label_matrix = np.array([[1]])


	def segment_image(self, img):
		segmented_img = np.zeros((1,img.shape[0]*img.shape[1]))
		img1 = np.reshape(img, (1,img.shape[0]*img.shape[1]*3))
		
		for i in range(int(img1.shape[1]/3)):
			pixel = img1[0,3*i:3*i+3]
			pixel = np.reshape(pixel, (3,1))
			classifier = self.omega.T @ pixel
			if classifier > 0:
				segmented_img[0,i] = 255
			else:
				segmented_img[0,i] = 0
	
		segmented_img = np.reshape(segmented_img, (img.shape[0], img.shape[1]))
		return segmented_img

		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		# YOUR CODE HERE

		
	def area_ratio(self, contour):
		'''
			Determine how much of a contour's area overlaps it's minimum enclosing circle

			Inputs:
				contour - a contour from a detected polygon
			Outputs:
				fraction of the contours area that overlaps with it's minimum enclosing circle
			# THIS CRITERIA HAS BEEN DEPRECATE AND IS NO LONGER USED FOR SHAPE DETECTION
		'''

		center, radius = cv2.minEnclosingCircle(contour)
		area_of_circle = math.pi * radius * radius
		area_of_contour = cv2.contourArea(contour)

		return area_of_contour/area_of_circle

	def likelihood_of_stop_sign(self, contour):

		'''
			Determine how close a contour matches an octagonal contour

			Inputs:
				contour - a contour from a detected polygon
			Outputs:
				similarity - a similarity score based on how the input contour resembles an octagon
			# THIS CRITERIA HAS BEEN DEPRECATE AND IS NO LONGER USED FOR SHAPE DETECTION
		'''
		
		# x_scale = img_size[0]/1390
		# y_scale = img_size[1]/866
		x_scale = 1
		y_scale = 1
		compare_to = np.array([[[int(478*x_scale),int(171*y_scale)]], 
								[[int(384*x_scale),int(174*y_scale)]], 
								[[int(315*x_scale),int(247*y_scale)]], 
								[[int(317*x_scale),int(341*y_scale)]], 
								[[int(390*x_scale),int(410*y_scale)]], 
								[[int(485*x_scale),int(407*y_scale)]], 
								[[int(553*x_scale),int(336*y_scale)]], 
								[[int(550*x_scale),int(237*y_scale)]]])
	
		
		
		#compare_to = compare_to*a1/a2
		similarity = cv2.matchShapes(compare_to,contour,3, 0.0)


		return similarity
	def ellipse_check(self, minor, major):

		'''
			Check how close a contour resembles a circle or ellipse

			Inputs:
				minor - minor axis of ellipse
				major - major axis of ellipse
			Outputs:
				pct_diff - percent difference between the major and minor axes
		'''

		pct_diff = (major-minor)/major
		return pct_diff

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the stop sign
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		boxes = []
		stop_signs = []
		mask_img = self.segment_image(img)
		mask_img = cv2.blur(mask_img,(2,2))
		#gray = cv2.cvtColor(np.uint8(mask_img), cv2.COLOR_GRAY2RGB)
		ret,thresh = cv2.threshold(mask_img,127,255,cv2.THRESH_BINARY)
		thresh2 = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)

		# The below 2 for loops are used to convert the thresholded image
		# into the proper format for cv2.findContours()
		# There were issues with the autograder when I used cv2.cvtColor() for this
		for i in range(thresh.shape[0]):
			for j in range(thresh.shape[1]):
				thresh2[i,j] = np.uint8(np.sum(thresh[i,j]/255))
		contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# approximating polygon contour
		for i in range(len(contours)):
			epsilon = 0.015*cv2.arcLength(contours[i],True)
			contours[i] = cv2.approxPolyDP(contours[i],epsilon,True)

		contours1 = sorted(contours, key=cv2.contourArea, reverse=True)
		# Taking largest contours
		cnt1 = contours1[:3]
		vertices = [7,8,9,10,11,12,13,14,15,16,17,18]
		
		# Main shape detection code
		for i in range(len(cnt1)):
			length = len(cnt1[i])
			
			# fitEllipse throws error if contour has length less than 5
			if length >4:
				(x, y), (minor, major), angle = cv2.fitEllipse(cnt1[i])
				value = self.ellipse_check(minor,major)
				if length in vertices and value < 0.25 and cv2.contourArea(cnt1[i]) > 200: 
					stop_signs.append(cnt1[i])
				
		
		# Creating list of xy coordinates for bounding box
		# x1,y1,x2,y2 are in cartesian form
		for i in range(len(stop_signs)):
			x,y,w,h = cv2.boundingRect(stop_signs[i])
			x1 = x
			y1 = img.shape[0]-y-h
			x2 = x+w
			y2 = img.shape[0]-y
			boxes.append([x1, y1, x2, y2])
			# boxes.append([x, y, x+h, y+h])

		# sorting boxes by their x1 coordinate
		boxes = sorted(boxes, key=lambda x: x[0])

		return boxes
		
		
	def obtain_data(self, img, mask):

		'''
			Obtain training data for an image and append it row wise to self.data_matrix and
			self.label_matrix
			
			Inputs:
				img - original image
				mask - binary mask for original image
			
		'''
		rows = img.shape[0]
		cols = img.shape[1]
		img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))

		#print(img.shape)
		mask = np.reshape(mask, (mask.shape[0]*mask.shape[1],1))
		
		mask = mask/255
		
		mask[mask == 0] = -1
		self.data_matrix = np.append(self.data_matrix, img, axis=0)
		self.label_matrix = np.append(self.label_matrix, mask, axis=0)

	def sigmoid(self,y, x, omega):

		'''
			Perform sigmoid calculation
			
			Inputs:
				y - scalar label of data point 1x1
				x - pixel vector 1x3
				omega 1x3
			
		'''

		y = np.reshape(y, (1,1))
		x = np.reshape(x, (1,3))
		val = y[0,0] * x @ omega.T
		
		# The math.exp hits math range errors if val is too negative
		# This solution to handle the exception was found
		# from: https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp

		if val[0] < 0:
			return 1 - 1/(1 + math.exp(val[0]))
		else:
			return 1/(1 + math.exp(-val[0]))

	def gradient_descent(self):
		'''
			Perform gradient descent and update learned parameters
			
		'''
		for i in range(self.iterations):
			print("Iteration: " + str(i+1))
			temp_omega = copy.deepcopy(self.omega)
			temp_gradient = copy.deepcopy(self.gradient)

			labels = self.label_matrix[1:,:]
			data = self.data_matrix[1:,:]
			for j in range(labels.shape[0]):
				sigmoid_val = self.sigmoid(y=labels[j,:], x=data[j,:], omega=self.omega)
				l = np.reshape(labels[j,:], (1,1))
				xval = np.reshape(data[j,:], (1,3))
				self.gradient = self.gradient + l[0,0]*xval*(1-sigmoid_val)

			self.omega = temp_omega + self.alpha*self.gradient
			delta_val = np.sum((temp_omega - self.omega))
			print("Omega: " + str(self.omega))
			self.delta_omega.append(delta_val)

if __name__ == '__main__':
	tfolder = "test_images"
	folder = "trainset"
	mask_folder = "masks"
	my_detector = StopSignDetector()
	counter = 1
	for filename in os.listdir(tfolder):
		if counter > 40:
			break
		counter = counter + 1
		print(str(filename))
		img = cv2.imread(os.path.join(os.getcwd(),tfolder,filename))
		#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		# cv2.imshow("img",img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		
		# mask = cv2.imread(os.path.join(os.getcwd(),mask_folder,filename), cv2.IMREAD_GRAYSCALE)
		# ret, thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
		# my_detector.obtain_data(img,thresh)

		
		boxes = my_detector.get_bounding_box(img)

		'''
		Below code is used for seeing bounding box results plotted
		on the original image. Make sure the returned box coordinates
		are in pixel coordinates. Currently they are set for cartesian
		coordinates due to the Autograder requested format. However,
		cv2.rectangle uses pixel coordinates.
		To switch to pixel coordinates:
		Comment this line --- > boxes.append([x1, y1, x2, y2])
		Uncomment this line --- > # boxes.append([x, y, x+h, y+h])
		'''
		# print("Stop Signs: " + str(len(boxes)))
		# for box in boxes:
		# 	x1 = box[0]
		# 	y1 = box[1]
		# 	x2 = box[2]
		# 	y2 = box[3]
		# 	cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), 2)
		# cv2.imshow("img", img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope
