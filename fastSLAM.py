import numpy as np
from scipy.stats import multivariate_normal
import json


class FastSlamParticle:
    def __init__(self):
        self.pos_prev = np.zeros(shape=[3])     # [x, y, theta]
        self.pos_curr = np.zeros(shape=[3])

        self.covR = np.diag(np.ones(shape=3))    # dummy covariance
        self.covQ = np.diag(np.ones(shape=3))    # dummy covariance

        # TODO: Check if the observed_feature carry on over different time step??
        self.observed_feature = list()  # store the id of landmark that is observed
        self.features_prev = list()      # list of tuple? (id, mu, sigma)
        self.features_curr = list()      # list of tuple? (id, mu, sigma)

        # TODO: Not sure if w is for single particle or 1 w for every feature?
        self.w = 0

    def get_importance_weight(self):
        return self.w

    # reset the observed feature when new one is created
    def _reset_observation(self):
        self.observed_feature = list()

    def sample_pose(self, action):

        v = action[0]
        theta = action[1]

        mean = self.pos_prev + np.array([
            [v * np.cos(self.pos_prev[2]) + theta],
            [v * np.sin(self.pos_prev[2]) + theta],
            [theta]
        ]).T

        self.pos_curr = np.random.multivariate_normal(np.squeeze(mean), self.covR)

    def update(self, measurements):    # measurement = [(id, dist, bearing), (id, dist, bearing)]

        _w = list()

        for measurement in measurements:

            id = measurement[0]  # find the id of the feature

            if id in self.observed_feature:

                mean_prev = self.features_prev[id]['mu']
                sigma_prev = self.features_prev[id]['sigma']

                # TODO: Get z_t from the measurement??
                z_t = np.array([measurement[0], measurement[1]])

                delta = mean_prev - self.pos_curr[:1]
                q = np.dot(delta.T, delta)

                # measurement prediction
                z_hat = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - self.pos_curr[2]]).T

                # Jacobian of H with respect to location, STILL HAVE TO MULTIPLE WITH F_BIG
                H = (1 / q) * np.array([
                    [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
                    [delta[1], -delta[0], -q, -delta[1], delta[0]]
                ])

                # measurement covariance
                covQ = np.dot(np.dot(H, sigma_prev), H.T) + self.covQ

                # kalman gain
                K = np.dot(np.dot(sigma_prev, H.T), np.linalg.inv(covQ))

                # update mean
                mean = mean_prev + K * (z_t - z_hat)

                # update covariance
                sigma = (np.eye(3) - K * H) * sigma_prev

                # importance factor
                w = multivariate_normal(mean, sigma).pdf(z_t)

                f = {'mu': mean,
                     'sigma': sigma,
                     'w': w
                     }
                self.features_curr.append(f)
                _w.append(w)        # record the importance weight of every landmark in every measurement?

            else:   # for those not observed

                rx = self.pos_curr[0]
                ry = self.pos_curr[1]
                rtheta = self.pos_curr[2]

                # landmark location
                mean = np.array([
                    [rx + measurement[1] * np.cos(rtheta + measurement[2])],
                    [ry + measurement[1] * np.sin(rtheta + measurement[2])]
                ])

                # difference between robot and landmark, delta[0] is x axis, delta[1] is y axis
                delta = mean - self.pos_curr[:1]

                q = np.dot(delta.T, delta)

                # Jacobian of H with respect to location, STILL HAVE TO MULTIPLE WITH F_BIG
                H = (1 / q) * np.array([
                    [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
                    [delta[1], -delta[0], -q, -delta[1], delta[0]]
                ])

                H_inv = np.linalg.inv(H)
                sigma = np.dot(np.dot(H_inv, self.covQ), H_inv.T)

                # default importance weight
                w = 1 / len(measurements)

                f = {'mu': mean,
                     'sigma': sigma,
                     'w': w
                     }

                self.observed_feature.append(id)
                self.features_curr.append(f)
                _w.append(w)

        for feature in self.features_prev:
            if feature['id'] not in self.observed_feature:  # not sure about this line
                self.features_curr.append(self.features_prev[feature['id']])

        self.pos_prev = self.pos_curr
        self.w = np.array(_w).mean()            # find the importance weight of this particle
        # reset the self.observed_feature??


def resample(w):
    w = np.array(w)
    M = w.shape[0]
    cumsum_w = np.cumsum(w)

    result_ind = np.zeros(shape=[M])

    for m in range(M):
        ind_m = np.where(cumsum_w > np.random.rand(1))[0]
        print ind_m
        if len(ind_m) != 0:
            result_ind[m] = ind_m[0]
        else:       # if no satisfied, select an ind randomly
            result_ind[m] = np.random.randint(0, M)

    return result_ind
