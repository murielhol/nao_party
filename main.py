import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from landmark_detection import landmark_detection
from utils import *


from fastSLAM import FastSlamParticle
from fastSLAM import resample


time = 10
num_particles = 1000

# initialize a set of particles
particle_set = [FastSlamParticle() for _ in range(num_particles)]

for t in range(time):

    # TODO: Input action, measurement here
    action = [0, 0]
    measurement = []

    w = list()

    for particle in particle_set:
        particle.sample_pose(action)
        particle.update(measurement)

        w_ = particle.get_importance_weight()
        w.append(w_)

    resampled_ind = resample(w)

    # select the particle of those indices from the set
    resampled_particle = [particle_set[ind] for ind in resampled_ind]

    # set of particle for next time step
    particle_set = resampled_particle
