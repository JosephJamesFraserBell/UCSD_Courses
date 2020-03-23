
#####################################
###### README for ECE 276A PR2 ######
###### 		Joseph Bell        ######
#####################################

Number of files: 2
File 1: PR2_code.py
File 2: depth_code.py

~~~~~~~~~~
 File 1 
~~~~~~~~~~

PR2_code.py
~~~~~~~~~~~

The code in this python script implements a particle filter approach to simultaneous
localization and mapping. There are 4 major steps in the code: prediction, updating, mapping, and resampling

The prediction step spans lines 274 to 315
----- The prediction step takes each particle and samples random noise from a Gaussian distribution with 0 mean
----- and a variance that is proportional to the change in pose between state t and t-1.
----- The approach to noise was to only add noise when the robot has moved.

The update step spans lines 323 to 346
----- The update state calculates the map correlation for each pixel and appends the correlation value to
----- a list so that it can be converted to a soft max later on. The map correlation is done with the 
----- mapCorrelation3 function. There is also a mapCorrelation2, but this is not used. mapCorrelation2 was
----- an attempt to rewrite the mapCorrelation function provided in the utility python script. mapCorrelation3
----- was written to overwrite mapCorrelation2 as it is quicker and does the same thing.

The mapping step spans lines 349 to 360
----- After the weights are converted to a soft max the largest weight is located and the corresponding
----- particle is used to update the log-odds map. The update_log_odds_map is responsible for updating the
----- log odds map.

The resamping step spans lines 363 to 370
----- Resampling is done via the resample function. The resample function uses monte carlo sampling from
----- the filterpy module. The number of effective particles is checked, and if it falls below the threshold
----- then the weights are reset to 1/N where N is the number of particles

Additional functions:

initializeMap -----> this function reads in the first lidar scan and updates the log odds map to start off the whole process

world_T_body_lidar -----> this function converts lidar scans in the lidar coordinate frame to the world coordinate frame

world_T_body_lidar2 -----> this is the vectorized version of world_T_body_lidar and is used because it is much quicker at
								converting between coordinate frames.



~~~~~~~~~~
 File 2 
~~~~~~~~~~

depth_code.py
~~~~~~~~~~~~~

This is my attempt at coloring the grid map using the RGB images. I was not successful and did
not include anything regarding this extra credit aspect of the project in my report. However, by the
off chance I get extra credit for attempting this I am submitting my code.