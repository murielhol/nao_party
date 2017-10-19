import numpy as np
import json


class FastSlamParticle:
    def __init__(self):
        self.pos_prev = np.zeros(shape=[3])
        self.pos_curr = np.zeros(shape=[3])

        # TODO: Define the covariance for sampling the pose
        self.cov = np.diag(np.ones(shape=3))    # dummy covariance
        self.A = np.zeros(shape=[3, 3])  # dummy variables, has to be defined
        self.B = np.zeros(shape=[3, 3])

        # TODO: Check if the observed_feature carry on over different time step??
        self.observed_feature = list()  # store the id of landmark that is observed
        self.features_prev = list()      # list of tuple? (id, mu, sigma)
        self.features_curr = list()      # list of tuple? (id, mu, sigma)

    def _sample_pose(self, action):
        # TODO: Find the mean according to equation 3.4??? combine action and previous pos???
        mean = np.dot(self.A, self.pos_prev) + np.dot(self.B, action)
        self.pos_curr = np.random.multivariate_normal(mean, self.cov)

    # TODO: Find out what should be the format of the features
    # probably a list of dict or list of tuple?
    def _update(self, new_features):
        # iterate through every features???

        for new_feature in new_features:

            id = new_feature[0]  # find the id of the feature

            if id in self.observed_feature:
                # find those features from self.features_prev
                # update those features
                # append into self.features_curr
                pass
            else:   # for those not observed
                # calculate mean, Jacobian, cov, w
                f = {'mu': 0,
                     'sigma': 0
                     }
                self.features_curr.append(f)

        for feature in self.features_prev:
            if feature['id'] not in self.observed_feature:  # not sure about this line
                # copy them from self.features_prev to self.features_curr
                pass

        # reset the self.observed_feature??

