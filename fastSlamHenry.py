import numpy as np
from scipy.stats import multivariate_normal
import json
import matplotlib.pyplot as plt

from read_data import read_data


def resample(w):
    w = np.array(w)
    # w = w / np.sum(w)

    M = w.shape[0]
    cumsum_w = np.cumsum(w)

    result_ind = np.zeros(shape=[M])

    for m in range(M):
        ind_m = np.where(cumsum_w > np.random.rand(1))[0]

        if len(ind_m) != 0:
            result_ind[m] = ind_m[0]
        else:       # if no satisfied, select an ind randomly
            result_ind[m] = np.random.randint(0, M)

    return result_ind


class Particle:
    def __init__(self, debug=False):
        self.debug = debug
        self.pose = np.zeros(shape=[3])
        self.features = dict()
        self.rot1 = 0

        self.features_prev = dict()
        self.features_curr = dict()

        self.covR = 0.0001 * np.eye(3)
        self.covQ = 0.0001 * np.eye(2)

        self.w = 0

    def sample_pose(self, u):
        v = u[0]
        theta = u[1]

        # predicted robot position mean
        x_new = np.array([
            self.pose[0] + v * np.cos(self.pose[2] + self.rot1),
            self.pose[1] + v * np.sin(self.pose[2] + self.rot1),
            self.pose[2] + theta
        ])

        if self.debug:
            print 'Estimated pose: {}'.format(x_new)
            print 'Original pose: {}'.format(self.pose)

        self.pose = np.random.multivariate_normal(np.squeeze(x_new), self.covR)

        if self.debug:
            print 'Sampled pose: {}'.format(self.pose)

    def update(self, measurement):
        _w = list()

        for i in range(6):      # iterate through 6 landmarks
            z_t = measurement[:, i]         # measurement for 1 landmark
            featureID = str(i)

            if z_t[0] == 0 and z_t[1] == 0:     # if no observation
                # if a feature is observed before, and not being observed now
                # leave that feature unchanged.
                if featureID in self.features_prev:
                    self.features_curr[featureID] = self.features_prev[featureID]
                continue

            if featureID in self.features_prev:     # observed feature
                mean = self.features_prev[featureID]['mu']
                sigma = self.features_prev[featureID]['sigma']

                # use mean_prev and current particle position to calculate r_hat and theta_hat
                delta = mean - self.pose[:2]

                q = np.dot(delta.T, delta)

                # measurement prediction
                z_hat = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0])]).T

                # Jacobian of H with respect to location
                # 2x2
                H = (1 / q) * np.array([
                    [np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
                    [-delta[1], delta[0]]
                ])

                # measurement covariance, 2x2
                Q = np.dot(np.dot(H, sigma), H.T) + self.covQ

                # kalman gain, 2x2
                K = np.dot(np.dot(sigma, H.T), np.linalg.inv(Q))

                # innovation
                inno = z_t - z_hat
                ro = np.dot(np.dot(inno, np.linalg.inv(Q)), inno)

                if ro < 2:
                    # update mean, 2x2
                    mean += np.dot(np.dot(K, (z_t - z_hat)), inno)

                    # update covariance, 2x2
                    sigma = np.dot((np.eye(2) - np.dot(K, H)), sigma)

                # importance factor
                w = multivariate_normal(z_hat, Q).pdf(z_t)

                self.features_curr[featureID]['mu'] = mean
                self.features_curr[featureID]['sigma'] = sigma
                _w.append(w)
            else:
                # initialize mean based on current particle position and measurement
                mean = np.array([
                    self.pose[0] + z_t[0] * np.cos(self.pose[2] + z_t[1]),
                    self.pose[1] + z_t[0] * np.sin(self.pose[2] + z_t[1])
                ]).T

                delta = mean - self.pose[:2]    # finding the delta x and delta y

                q = np.dot(delta.T, delta)      # finding the distance square

                # 2x2
                H = (1 / q) * np.array([
                    [np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
                    [-delta[1], delta[0]]
                ])

                # 2x2
                H_inv = np.linalg.inv(H)

                # 2x2
                sigma = np.dot(np.dot(H_inv, self.covQ), H_inv.T)

                # default importance weight
                w = 0.00001

                f = {
                     'mu': mean,
                     'sigma': sigma
                     }

                self.features_curr[featureID] = f
                _w.append(w)

        self.features_prev = self.features_curr
        self.w = np.array(_w).sum()


u, x, z = read_data()
print u.shape, x.shape, z.shape

num_particles = 100

particles = [Particle() for _ in range(num_particles)]

for t in range(140, 301):

    w = list()

    print '======= t = {} ======='.format(t)

    for i in range(num_particles):
        particles[i].sample_pose(u[:, t])
        # particles[i].update(z[:, :, t])

        # w.append(particles[i].w)

    # resampled_ind = resample(w)

    # select the particle of those indices from the set
    # resampled_particle = [particles[int(ind)] for ind in resampled_ind]

    # set of particle for next time step
    # particles = resampled_particle

    if t % 100 == 0:
        xmean = [j.pose[0] for j in particles]
        ymean = [j.pose[1] for j in particles]

        plt.figure()
        plt.scatter(xmean, ymean)
        plt.show()

        if t == 300:
            print '####################'
            w = list()
            for i in range(num_particles):
                pose = particles[i].pose
                print pose[:2]
                w_tmp = multivariate_normal(pose[:2], np.array([[1, 0], [0, 1]])).pdf(np.array([2.5, -3.5]))
                print w_tmp
                # print w_tmp
                w.append(w_tmp)

            resampled_ind = resample(w)
            resampled_particle = [particles[int(ind)] for ind in resampled_ind]
            particles = resampled_particle

            xmean = [j.pose[0] for j in particles]
            ymean = [j.pose[1] for j in particles]

            plt.figure()
            plt.scatter(xmean, ymean)
            plt.show()

