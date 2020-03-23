'''
ECE276A WI20 HW1
Stop Sign Detector
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
		self.iterations = 40
		#self.omega = np.array([[0, 0, 0]])
		# self.omega = np.array([[15944646.7],
		#  					   [-80957937.05],
		#  					   [56327290.0]])
		#self.omega = np.array([[2.05453730e+08],[-8.99771245e+08],[-7.92851682e+08]])
		self.omega = np.array([[1],[-1.082754],[-0.982788354]])
		self.delta_omega = []
		self.likelihood_model = 0
		self.alpha = 0.1
		self.gradient = np.array([[0, 0, 0]])
		self.data_matrix = np.array([[0,0,0]])
		self.label_matrix = np.array([[1]])

	def segment_image(self, img):
		#segmented_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
		segmented_img = np.zeros((1,img.shape[0]*img.shape[1]))
		#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img1 = np.reshape(img, (1,img.shape[0]*img.shape[1]*3))
		print(img1.shape)
		for i in range(int(img1.shape[1]/3)):
			pixel = img1[0,3*i:3*i+3]
			pixel = np.reshape(pixel, (3,1))
			classifier = self.omega.T @ pixel
			if classifier > 0:
				segmented_img[0,i] = 255
			else:
				segmented_img[0,i] = 0
		# for i in range(img.shape[0]):
		# 	for j in range(img.shape[1]):
		# 		pixel = img[i,j]
		# 		classifier = self.omega.T @ pixel

		# 		if classifier > 0:
		# 			segmented_img[i,j] = (255,255,255)
		# 		else:
		# 			segmented_img[i,j] = (0,0,0)
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
		center, radius = cv2.minEnclosingCircle(contour)
		area_of_circle = math.pi * radius * radius
		area_of_contour = cv2.contourArea(contour)

		return area_of_contour/area_of_circle
	def likelihood_of_stop_sign(self, contour):
		
		# x_scale = img_size[0]/1390
		# y_scale = img_size[1]/866
		x_scale = 0.5
		y_scale = 0.5
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
		#pct_diff = (np.sqrt((major/2)**2-(minor/2)**2))/(major/2)
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
		mask_img = self.segment_image(img)
		# cv2.imshow('mask',mask_img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		boxes = []
		#mask_img = cv2.GaussianBlur(mask_img,(3,3),0)
		mask_img = cv2.blur(mask_img,(2,2))
		# print(img.shape)

		# gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGB)
		# cv2.imshow('mask',mask_img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# plt.imshow(img, cmap='gray')
		# plt.show()
		
		ret,thresh = cv2.threshold(mask_img,127,255,cv2.THRESH_BINARY)

		# cv2.imshow('mask',thresh)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# plt.imshow(thresh)
		# plt.show()


		
		
		
		thresh2 = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)
		for i in range(thresh.shape[0]):
			for j in range(thresh.shape[1]):
				thresh2[i,j] = np.uint8(np.sum(thresh[i,j]/255))

		
		# plt.imshow(thresh2, cmap='gray')
		# plt.show()
		contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		for i in range(len(contours)):
			epsilon = 0.015*cv2.arcLength(contours[i],True)
			contours[i] = cv2.approxPolyDP(contours[i],epsilon,True)

		contours1 = sorted(contours, key=cv2.contourArea, reverse=True)
		contours2 = sorted(contours, key=self.area_ratio, reverse=True)
		contours3 = sorted(contours, key=self.likelihood_of_stop_sign, reverse=False)
		
		
		# cnt1 = contours1[:20]
		# cnt3 = sorted(cnt2, key=self.area_ratio, reverse=True)
		# cnt1 = cnt3[:int(len(cnt3)/2)]
		cnt1 = contours1[:8]
		cnt2 = contours2[:10]
		cnt3 = contours3[:8]
		# for con in cnt1:
		# 	print(self.area_ratio(con))
		# print("------------------------------------------")
		# for con in cnt2:
		# 	print(self.area_ratio(con))
		# print("------------------------------------------")
		# for con in cnt1:
		# 	print(self.likelihood_of_stop_sign(con, img.shape))
			
		# cv2.drawContours(img, cnt1, -1, (0,0,255), 3)
		# cv2.imshow('contours',img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# cv2.drawContours(gray, cnt2, -1, (0,255,0), 3)
		# cv2.imshow('contours',gray)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		vertices = [7,8,9,10,11,12]
		stop_signs = []
		for i in range(len(cnt1)):
			found = False
			length = len(cnt1[i])
			for j in range(len(cnt2)):
				if cv2.contourArea(cnt1[i] == cv2.contourArea(cnt2[j])):
					for k in range(len(cnt3)):
						if cv2.contourArea(cnt1[i] == cv2.contourArea(cnt3[k])):
							found = True
							break
			if length >= 7 and found == True: 
				stop_signs.append(cnt1[i])
				
			
		for i in range(len(stop_signs)):
			x,y,w,h = cv2.boundingRect(stop_signs[i])
			x1 = x
			y1 = img.shape[0]-y-h
			x2 = x+w
			y2 = img.shape[0]-y
			# print(str(x1) + "," + str(y1))
			# print(str(x2) + ", " + str(y2))
			boxes.append([x1, y1, x2, y2])
			#boxes.append([x, y, x+h, y+h])

		boxes = sorted(boxes, key=lambda x: x[0])
		print(boxes)
		#print(len(boxes))

		return boxes
		
		

	def obtain_data(self, img, mask):
		rows = img.shape[0]
		cols = img.shape[1]
		img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))

		#print(img.shape)
		mask = np.reshape(mask, (mask.shape[0]*mask.shape[1],1))
		
		mask = mask/255
		
		mask[mask == 0] = -1
		self.data_matrix = np.append(self.data_matrix, img, axis=0)
		self.label_matrix = np.append(self.label_matrix, mask, axis=0)

		print(self.data_matrix.shape)
		print(self.label_matrix.shape)

	def sigmoid(self,y, x, omega):
		y = np.reshape(y, (1,1))
		x = np.reshape(x, (1,3))
		val = y[0,0] * x @ omega.T
		# print(val.shape)
		# print(val)
		if val[0] < 0:
			return 1 - 1/(1 + math.exp(val[0]))
		else:
			return 1/(1 + math.exp(-val[0]))

	def gradient_descent(self):
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
			delta_val = np.sum(abs(temp_omega - self.omega))
			print("Delta: " + str(delta_val))
			self.delta_omega.append(delta_val)

if __name__ == '__main__':
	tfolder = "test_images"
	folder = "trainset"
	mask_folder = "masks"
	my_detector = StopSignDetector()
	counter = 1
	for filename in os.listdir(tfolder):
		# if counter > 6:
		# 	break
		counter = counter + 1
		print(str(filename))
		img = cv2.imread(os.path.join(os.getcwd(),tfolder,filename))
	# 	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# 	mask = cv2.imread(os.path.join(os.getcwd(),mask_folder,filename), cv2.IMREAD_GRAYSCALE)
	# 	ret, thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
	# 	my_detector.obtain_data(img,thresh)
	# my_detector.gradient_descent()
	# print(my_detector.omega)
	# print("\n\n\n")
	# print(my_detector.delta_omega)
	# # 	#Display results:
	# # 	#(1) Segmented images
		
	# 	mask_img = np.uint8(mask_img)
	# 	# plt.imshow(mask_img, cmap='gray')
	# 	# plt.show()
	# 	#(2) Stop sign bounding box
		
		boxes = my_detector.get_bounding_box(img)

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
		# for box in boxes:
		# 	x1 = box[0]
		# 	y1 = box[1]
		# 	x2 = box[2]
		# 	y2 = box[3]

		# 	plt.plot(x1, y1, 'bo')
		# 	plt.plot(x2, y2, 'go')
		# 	plt.show()
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope
