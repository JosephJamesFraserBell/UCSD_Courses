import numpy as np
import matplotlib.pyplot as plt

#############################
# CALCULATIONS FOR NUMBER 1d #
#############################

p = np.array([[-1],[1],[0]])
world_coordinates = np.array([[1],[5],[-2],[1]])

q0 = np.sqrt(3)/2
q1 = 0
q2 = 0
q3 = 1/2

R11 = 1 - 2*(q2**2 + q3**2)
R12 = 2*(q1*q2-q0*q3)
R13 = 2*(q0*q2 + q1*q3)
R21 = 2*(q1*q2 + q0*q3)
R22 = 1 - 2*(q1**2+q3**2)
R23 = 2*(q2*q3 - q0*q1)
R31 = 2*(q1*q3 - q0*q2)
R32 = 2*(q0*q1 + q2*q3)
R33 = 1 - 2*(q1**2 + q2**2)

Rotation_Matrix = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])

R_or = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])
M11 = R_or@Rotation_Matrix.T
M12 = -1*R_or@Rotation_Matrix.T@p
M = np.concatenate((M11,M12), axis=1)
M = np.concatenate((M, np.array([[0,0,0,1]])), axis=0)
optical_coordinates = M@world_coordinates
print(optical_coordinates)
K = np.array([[0.2*10,0.2*0,160.5],[0,0.2*10,120.5],[0,0,1]])
pixel_coordinates = K@(1/optical_coordinates[2,0]*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))@optical_coordinates
print(pixel_coordinates)


#############################
# CALCULATIONS FOR NUMBER 2 #
#############################

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
			states_moved[24] = states_moved[24] + states[x]
			# continue

		elif new_index < 0:
			states_moved[0] = states_moved[0] + states[x]
			# continue
		else:
			states_moved[new_index] = states[x]
		# states_moved[new_index] = states_moved[new_index] + states[x]
		# states_moved[new_index] = states[x]

		


	if u < 0:
		conv_kernel = [1/6, 1/2, 1/3]
	elif u > 0:
		conv_kernel = [1/3, 1/2, 1/6]
	elif u == 0:
		conv_kernel = [0, 1, 0]

	
	prediction = np.convolve(conv_kernel, states_moved, 'same')
	if u < 0:
		prediction[0] = prediction[0] + states_moved[0]*1/3
		prediction[24] = prediction[24] + states_moved[24]*1/6
	elif u > 0:
		prediction[0] = prediction[0] + states_moved[0]*1/6
		prediction[24] = prediction[24] + states_moved[24]*1/3


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
solution_grid = np.zeros((10,25))
state = states_init
for j in range(5):
	updated_state = update_step(z=observations[j], states=state)
	state = updated_state / np.sum(updated_state)
	
	for i in range(25):
		solution_grid[j*2,i] = round(state[i], 4)
	state = prediction_step(u=controls[j], states=state)
	
	for i in range(25):
		solution_grid[j*2+1,i] = round(state[i], 4)
normalize_state = state/np.sum(state)

plt.bar(np.linspace(0,25,25), normalize_state, width=1.0)
plt.title("Position distribution at t=4")
plt.show()


