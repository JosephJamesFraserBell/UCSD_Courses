import numpy as np
import matplotlib.pyplot as plt
## calculations for number 1
# p = np.array([[-1],[1],[0]])
# world_coordinates = np.array([[1],[2],[3],[1]])

# q0 = np.sqrt(3)/2
# q1 = 0
# q2 = 0
# q3 = 1/2

# R11 = 1 - 2*(q2**2 + q3**2)
# R12 = 2*(q1*q2-q0*q3)
# R13 = 2*(q0*q2 + q1*q3)
# R21 = 2*(q1*q2 + q0*q3)
# R22 = 1 - 2*(q1**2+q3**2)
# R23 = 2*(q2*q3 - q0*q1)
# R31 = 2*(q1*q3 - q0*q2)
# R32 = 2*(q0*q1 + q2*q3)
# R33 = 1 - 2*(q1**2 + q2**2)

# Rotation_Matrix = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])

# R_or = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])
# M11 = R_or@Rotation_Matrix.T
# M12 = -R_or@Rotation_Matrix.T@p
# M = np.concatenate((M11,M12), axis=1)
# M = np.concatenate((M, np.array([[0,0,0,1]])), axis=0)
# optical_coordinates = M@world_coordinates

# K = np.array([[0.2*10,0.2*0,160.5],[0,0.2*10,120.5],[0,0,1]])
# pixel_coordinates = K@(1/optical_coordinates[2,0]*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))@optical_coordinates
# print(pixel_coordinates)

# calculations for number 2



def prob_z_given_x(z, x_index):
	front_of_door = [7, 12, 16]
	near_door = [6, 8, 11, 13, 15, 17]
	prob_z = 0
	if x_index in front_of_door: 
		prob_z = 0.9
	elif x_index in near_door: 
		prob_z = 0.7
	else:
		prob_z = 0.20

	if z == -1:
		prob_z = 1 - prob_z

	return prob_z


def prediction_step(u ,states):
	# shift data
	states_moved = [0]*25
	states_predicted = [0]*25
	for x in range(25):
		new_index = x + u
		if new_index > 24:
			new_index = 24
		elif new_index < 0:
			new_index = 0
		# states_moved[new_index] = states_moved[new_index] + states[x]
		states_moved[new_index] = states[x]

	# sgn_u = 0
	# if u > 0:
	# 	sgn_u = 1
	# elif u < 0:
	# 	sgn_u = -1

	
	# for x in range(25):
	# 	move1 = x + u
	# 	move2 = x + u + sgn_u
	# 	move3 = x + u - sgn_u
	# 	if move1 > 24:
	# 		move1 = 24
	# 	elif move1 < 0:
	# 		move1 = 0
	# 	if move2 > 24:
	# 		move2 = 24
	# 	elif move2 < 0:
	# 		move2 = 0
	# 	if move3 > 24:
	# 		move3 = 24
	# 	elif move3 < 0:
	# 		move3 = 0
	# 	px11 = states_moved[move1] * 1/2
	# 	px12 = states_moved[move2] * 1/3
	# 	px13 = states_moved[move3] * 1/6
	# 	prediction = px11 + px12 + px13
	# 	states_predicted[x] = prediction

	if u > 0:
		conv_kernel = [1/6, 1/2, 1/3]
	elif u < 0:
		conv_kernel = [1/3, 1/2, 1/6]
	elif u == 0:
		conv_kernel = [0, 1, 0]

	prediction = np.convolve(conv_kernel, states_moved, 'same')

	return prediction

def update_step(z, states):
	states_update = states.copy()
	for i in range(25):
		p_z = prob_z_given_x(z, x_index=i)*states[i]
		states_update[i] = p_z
	return states_update

states_init= [1/25]*25
observations = [1, -1, -1, -1, 1]
controls = [1, -1, 2, 1, 1]

prediction_steps = states_init
for j in range(5):
	update_steps = update_step(z=observations[j], states=prediction_steps)
	prediction_steps = update_steps / np.sum(update_steps)
	prediction_steps = prediction_step(u=controls[j], states=prediction_steps)

normalize_prediction_steps = prediction_steps/np.sum(prediction_steps)
plt.bar(np.linspace(0,25,25), normalize_prediction_steps, width=1.0)
plt.title("Position Prediction at t=0")
plt.show()
