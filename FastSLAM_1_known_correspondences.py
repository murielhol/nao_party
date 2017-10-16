import numpy as np
import math
	
def FastSLAM_1_known_correspondences_step(z_t, c_t, u_t, Y_t_1, observed_features):

	# initialise Q_t
	variance_err = .1
	Q_t = variance_err*np.eye(3)

	weights = np.ones(len(Y_t_1),6)

	# Loop over all particles in particleswarm
	for iterator in range(0,Y_t_1.shape[0]):
		particle = Y_T_1[iterator]
		# Retrieve location and bearing of the particle
		x_t_1 = particle[0]



		# Sample the new location and bearing with the help of the motion model and the previous location.
		x_t = sample(x_t_1, u_t)

		# For every observed feature.
		for j in c_t:
			if not(j in observed_features):
				observed_features.append(j)

				# Calculate mean
				mean = 

				# Calculate H
				H = 

				# Calculation the covariance
				covariance = np.multiply(np.multiply(np.inv(H),Q_t),np.transpose(np.inv(H))) 

				# Insert new Landmark
				new_landmark = [mean, covariance]
				particle[j] = new_landmark

				#Update Weights
				weights[iterator] = np.ones(6)
			else:
				# Extract Landmark location
				landmark = particle[j]

				# Measurement prediction
				z_measurement = 

				# Calculate Jacobian
				H = 

				# Measurement Covariance
				Q = np.multiply(np.multiply(H,landmark[1]),np.transpose(H)) + Q_t

				# Calculate Kalman gain
				K = np.multiply(np.multiply(landmark[1],np.transpose(H)),Q)
				
				# update mean
				mean = landmark[0] + np.multiply(K,(z_t-z_measurement))

				# Update Covariance
				covariance = np.multiply((np.eye(np.shape(K)[0]) - np.multiply(K,H)),landmark[1]) 

				# update Weights
				weights[iterator] = np.power(np.det(2*math.pi*Q),-0.5)*np.exp(0.5*(np.transpose(z_t - z_measurement))*np.inv(Q)*(z_t - z_measurement))


	Y_t = np.zeros(Y_t_1.shape())

	for iterator in range(0,Y_t_1.shape[0]):

		# Sample new Y_t
		Y_t[iterator] = sample_Y(Y_t_1[iterator])









	return Y_t






def sample(x_t_1, u_t):

	x_t = x_t_1

	return x_t