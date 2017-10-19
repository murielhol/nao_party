import numpy as np
from scipy.stats import multivariate_normal
import json


class FastSlamParticle:
    def __init__(self):
        self.pos_prev = np.zeros(shape=[3])     # [x, y, theta]
        self.pos_curr = np.zeros(shape=[3])

        self.covR = np.diag(np.ones(shape=3))    # dummy covariance
        self.covQ = np.diag(np.ones(shape=2))    # dummy covariance

        # TODO: Check if the observed_feature carry on over different time step??
        self.observed_feature = list()  # store the id of landmark that is observed
        self.features_prev = dict()      # dict of dict {id: {mu: 123, sigma:1}}
        self.features_curr = dict()

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

        num_lm = measurements.shape[1]      # number of landmarks

        _w = list()     # importance weight for the particle, (mean of weight calculated for all the features)

        # iterate through all the measurement
        for i in range(num_lm):
            measurement = measurements[:, i]

            # skip the features not being measured
            if measurement[0] == 0 and measurement[1] == 0:
                continue

            id = i

            if id in self.observed_feature:
                # print 'observed feature'

                mean_prev = self.features_prev[str(id)]['mu']
                sigma_prev = self.features_prev[str(id)]['sigma']

                z_t = np.array([measurement[0], measurement[1]])

                # use mean_prev and current particle position to calculate r_hat and theta_hat
                delta = mean_prev - self.pos_curr[:2]

                q = np.dot(delta.T, delta)

                # measurement prediction
                z_hat = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - self.pos_curr[2]]).T

                # Jacobian of H with respect to location, STILL HAVE TO MULTIPLE WITH F_BIG
                # 2x2
                H = (1 / q) * np.array([
                    [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1]],
                    [delta[1], -delta[0]]
                ])

                # measurement covariance, 2x2
                covQ = np.dot(np.dot(H, sigma_prev), H.T) + self.covQ

                # kalman gain, 2x2
                K = np.dot(np.dot(sigma_prev, H.T), np.linalg.inv(covQ))

                # update mean, 2x2
                mean = mean_prev + np.dot(K, (z_t - z_hat))

                # update covariance, 2x2
                sigma = np.dot((np.eye(2) - np.dot(K, H)), sigma_prev)

                # importance factor
                w = multivariate_normal(mean, sigma).pdf(z_t)

                f = {
                     'mu': mean,
                     'sigma': sigma
                     }

                self.features_curr[str(id)] = f
                _w.append(w)        # record the importance weight of every landmark in every measurement?

            else:   # for those not observed
                # print 'not observed feature'

                rx = self.pos_curr[0]
                ry = self.pos_curr[1]
                rtheta = self.pos_curr[2]

                # landmark location
                mean = np.array([
                    rx + measurement[0] * np.cos(rtheta + measurement[1]),
                    ry + measurement[0] * np.sin(rtheta + measurement[1])
                ]).T

                # difference between robot and landmark, delta[0] is x axis, delta[1] is y axis
                delta = mean - self.pos_curr[:2]

                q = np.dot(delta.T, delta)

                # 2x2
                H = (1 / q) * np.array([
                    [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1]],
                    [delta[1], -delta[0]]
                ])

                # 2x2
                H_inv = np.linalg.inv(H)

                # 2x2
                sigma = np.dot(np.dot(H_inv, self.covQ), H_inv.T)

                # default importance weight
                w = 1.0 / measurements.shape[1]

                f = {
                     'mu': mean,
                     'sigma': sigma
                     }

                self.observed_feature.append(id)
                self.features_curr[str(id)] = f
                _w.append(w)

        for feature_id, val in self.features_prev.iteritems():
            if feature_id not in self.observed_feature:  # not sure about this line
                self.features_curr[str(feature_id)] = val

        # copy the calculated feature, pos to '*_prev', so '*_curr' can be override in next iteration
        self.pos_prev = self.pos_curr
        self.features_prev = self.features_curr

        # find the importance weight of this particle
        self.w = np.array(_w).mean() if len(_w) > 0 else 0
        # reset the self.observed_feature??


def resample(w):
    w = np.array(w)
    M = w.shape[0]
    cumsum_w = np.cumsum(w)

    result_ind = np.zeros(shape=[M])

    for m in range(M):
        ind_m = np.where(cumsum_w > np.random.rand(1))

        if len(ind_m) != 0 and len(ind_m[0]) != 0:
            result_ind[m] = ind_m[0][0]
        else:       # if no satisfied, select an ind randomly
            result_ind[m] = np.random.randint(0, M)

    return result_ind
