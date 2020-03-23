Joseph Bell
2/2/2020
ECE276A PR1

This project consists of 2 python files: stop_sign_detector.py and prac.py

(prac.py is short for practice.py -- this file was also used to double check
matrix math and other small code snippets, but those were erased as they're not
really necessary for submission. Relevant code was kept. )


These files are used for Project 1 of ECE276A at UCSD. 


File 1: stop_sign_detector.py

This python file uses learned parameters from a probabilistic color model for 
binary color segmentation of an input image. Learned parameters are first obtained by
resetting the self.omega variable to [0,0,0] and running the obtain_data function.
The obtain_data function appends input images and binary masks to an instance variable
storing all the data from all training images. The obtain_data functions works for 1 file
and its corresponding mask at a time - so it must be ran inside a for loop (i.e. obtain_data
is called once per training image). Regarding the binary mask, white pixels are converted to
a scalar label of 1 and black pixels are converted to a scalar label of -1.

Outside the for loop one must run the gradient_descent function (only needs to be called once
after all training data is stored in the data_matrix and label_matrix variables). This
function iterates for however many iterations the self.iterations variable is set to. 

After gradient descent is completed - a learned set of parameters is found that is then
used to overwrite the self.omega variable. Now with the learned parameters, the get_bounding_box
function can be used. The obtain_data and gradient_descent functions are no longer needed.
The get_bounding_box functions works once per test image so it must be inside a for loop.
The get_bounding_box function calls the segment_image function which uses the learned
parameters to perform binary color segmentation on the input image. The color segmentation
used for this project was red vs non-red. Therefore, the segmented image is a binary mask 
where white pixels correspond to red pixels in the original image and black pixels correspond 
to non-red pixels.

As the goal of this project is to detect stop signs - shape detection analysis is performed on color segmented
mask to determine which white masses of pixels are "stop sign shaped". If a group of white pixels passes the 
criteria set for a stop sign shape (i.e. it has a percent difference between its major and minor axes that is 
less than 25 percent and the  number of edges of it's contour are within the acceptable range of 7 and 18) 
then that contour is deemed to be a stop sign and a bounding box is created for it. The coordinates
for the bounding box are in cartesian coordinates.
To return the bounding box coordinates in pixel coordinates:
Comment this line ---- > boxes.append([x1, y1, x2, y2])
Uncomment this line ---- > # boxes.append([x, y, x+h, y+h])

Note: by simply modifying the shape detection criteria this code can easily be used for detecting other
objects.

File 2: prac.py

This file consists of code used for generating the convergence plot and color space
plots for the project report. The code used for hand labeling the training data using 
roipoly is also included (https://github.com/JCardenasRdz/roipoly.py). Roipoly is
also referenced in the report.

Code for color space plots was sampled from: https://realpython.com/python-opencv-color-spaces/