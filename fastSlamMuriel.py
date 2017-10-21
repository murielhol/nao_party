
import numpy as np
from scipy.stats import multivariate_normal
import json


R = np.diag(np.ones(shape=2)/10)    # dummy covariance
Q = np.diag(np.ones(shape=2)/10)    # dummy covariance

class FastSlamParticle:
    def __init__(self):
        self.pose = np.zeros(shape=[3])
        self.features = dict() # dict of dict {id: {mu: 123, sigma:1}}
        self.w = 0
        self.rot1 = 0
        self.mean = 0
        self.cov = np.zeros((3,3))
        self.observed = []

    def sample_pose(self, action):
            v = action[0]
            theta = action[1]
            # predicted robot position mean
            self.mean = np.array([
                self.pose[0] + v * np.cos(self.pose[2] + self.rot1),
                self.pose[1] + v * np.sin(self.pose[2] + self.rot1),
                self.pose[2] + theta
            ])

            delta = self.pose - self.mean
            self.rot1 = np.arctan2(delta[1], delta[0]) - self.pose[2]

            # jacobian, robot position
            G = np.array([
                [1, 0, -v * np.sin(self.pose[2] + self.rot1)],
                [0, 1, v * np.cos(self.pose[2] + self.rot1)],
                [0, 0, 1]
            ])
            # jacobian, control
            V = np.array([
                [np.cos(self.pose[2] + self.rot1), - v * np.sin(self.pose[2] + self.rot1)],
                [np.sin(self.pose[2] + self.rot1), v * np.cos(self.pose[2] + self.rot1)],
                [0, 1]
            ])
            # predicted covariance
            self.cov = np.dot(np.dot(G, self.cov), G.T) + np.dot(np.dot(V, R), V.T)
            # sample new pose
            self.pose = np.random.multivariate_normal(self.mean, self.cov)

    def update(self, measurements):

        num_lm = measurements.shape[1]      # number of landmarks

        _w = []   # importance weight for the particle, (mean of weight calculated for all the features)

        # iterate through all the measurement
        for i in range(num_lm):
            measurement = measurements[:, i]

            # skip the features not being measured
            if measurement[0] == 0 and measurement[1] == 0:
                continue

            id = i

            if not(id in self.observed):
                mean = np.zeros(shape=[2])
                sigma = np.eye(2)/100

                f = {
                     'mu': mean,
                     'sigma': sigma
                     }
                self.features[str(id)] = f
                # self.features[str(id)]['mu'] = np.zeros((2,1))
                # self.features[str(id)]['sigma'] = np.zeros((2,2))
                self.observed.append(id)



            if id in self.observed:
                mean = self.features[str(id)]['mu']
                sigma = self.features[str(id)]['sigma']
                z_t = measurement[:2]
                # use mean and current particle position to calculate r_hat and theta_hat
                delta = mean - self.pose[:2]

                q = np.dot(delta.T, delta)
                # measurement prediction
                z_hat = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - self.pose[2]]).T

                h = (mean[0]**2) - 2*mean[0]*self.pose[0]+(self.pose[0]**2)+(mean[1]**2)-2*self.pose[1]*self.pose[1]+(self.pose[1]**2)
                H = np.array([
                    [-(mean[0]-self.pose[0])/np.sqrt(h), -(mean[1]-self.pose[1])/np.sqrt(h)],
                    [(mean[1]-self.pose[1])/h,  -(mean[0]-self.pose[0])/h]
                    ])
                S = np.dot(np.dot(H, sigma), H.T) + Q

                K = np.divide(np.dot(sigma, H).T,S)
                nu = z_t - z_hat

                ro = np.dot(np.dot(nu, np.linalg.inv(S)), nu.T)


                if ro < 2:
                    mean = mean + np.dot(K, nu)
                    sigma = np.dot((np.eye(2) - np.dot(K, H)), sigma)

                # importance factor

                w = multivariate_normal(z_hat, Q).pdf(z_t)

                f = {
                     'mu': mean,
                     'sigma': sigma
                     }

                self.features[str(id)] = f
                _w.append(w)        # reco

            self.w = np.sum(np.array(_w)) if len(_w) > 0 else 0



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
 

