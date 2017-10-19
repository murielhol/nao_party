import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_detection import landmark_detection
from utils import *


from fastSLAM import FastSlamParticle
from fastSLAM import resample
from read_data import read_data


num_particles = 1000

# load the data
u, x, z = read_data()

time = u.shape[1]

# experiment with 1 particle #####
p1 = FastSlamParticle()
p1.sample_pose(u[:, 0])
p1.update(z[:, :, 0])
# print '##### Updated... back to main function ... #####'
# print 'importance weight: {}'.format(p1.get_importance_weight())

p1.update(z[:, :, 0])
# print '##### Updated... back to main function ... #####'
# print 'importance weight: {}'.format(p1.get_importance_weight())



# initialize a set of particles
# particle_set = [FastSlamParticle() for _ in range(num_particles)]

# for t in range(time):
#     print 'time: {}'.format(t)
#
#     # data at the current time step
#     u_t = u[:, t]
#     x_t = x[:, t]
#     z_t = z[:, :, t]
#
#     w = list()
#
#     for i, particle in enumerate(particle_set):
#         particle.sample_pose(u_t)
        # particle.update(z_t)

        # w_ = particle.get_importance_weight()
        # w.append(w_)

    # resampled_ind = resample(w)
    #
    # # select the particle of those indices from the set
    # resampled_particle = [particle_set[int(ind)] for ind in resampled_ind]
    #
    # # set of particle for next time step
    # particle_set = resampled_particle
