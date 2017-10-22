import numpy as np
from scipy.stats import multivariate_normal
import json
import matplotlib.pyplot as plt

from read_data import read_data


def normal_density(x, mu, sigma):
    return np.exp(-.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu)))/np.sqrt(2*np.pi*np.linalg.det(sigma))


def resample(w):

    w = np.array(w)
    w = w / np.sum(w)
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
        self.rot1 = 0

        self.features_prev = dict()
        self.features_curr = dict()

        self.covR = 0.00001 * np.eye(3)
        self.covQ = 0.1 * np.eye(2)

        self.w = 0

    def sample_pose(self, u):

        # predicted robot position mean
        x_new = find_next_pos(self.pose, u, self.rot1, self.covR)

        self.pose = x_new
        # self.rot1 = self.pose[2].copy()

        if self.debug:
            print 'Sampled pose: {}'.format(self.pose)

    def update(self, measurement):
        _w = list()

        for i in range(6):      # iterate through 6 landmarks
            z_t = measurement[:, i]         # measurement for 1 landmark
            featureID = str(i)
            w = 1

            if z_t[0] == 0 and z_t[1] == 0:     # if no observation
                # if a feature is observed before, and not being observed now
                # leave that feature unchanged.
                if featureID in self.features_prev:
                    self.features_curr[featureID] = self.features_prev[featureID].copy()
                continue

            if featureID in self.features_prev:     # observed feature
                mean = self.features_prev[featureID]['mu']
                sigma = self.features_prev[featureID]['sigma']

                # use mean_prev and current particle position to calculate r_hat and theta_hat
                delta = mean - self.pose[:2]

                q = np.dot(delta.T, delta)

                # measurement prediction
                z_hat = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) + self.pose[2]]).T

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
                    w = normal_density(inno, np.zeros(2), Q)
                else:
                    w = 0.0000000001

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
                Q = np.dot(np.dot(H, sigma), H.T) + self.covQ
                w = normal_density(0, np.zeros(2), Q)

                f = {
                     'mu': mean,
                     'sigma': sigma
                     }

                self.features_curr[featureID] = f
                _w.append(w)

        self.features_prev = self.features_curr.copy()
        self.w = np.array(_w).sum()


def find_next_pos(pos, u, rot1, sigma):

    mean = np.array([
        pos[0] + u[0] * np.cos(pos[2] + rot1),
        pos[1] + u[0] * np.sin(pos[2] + rot1),
        pos[2] + u[1]
    ])
    pos = np.random.multivariate_normal(np.squeeze(mean), sigma)

    return pos


u, x, z = read_data()
print u.shape, x.shape, z.shape

Lx = [-15, -15, 0, 0, 15, 15]
Ly = [-10, 10, -10, 10, -10, 10]

num_particles = 100

particles = [Particle() for _ in range(num_particles)]


plt.style.use('ggplot')
plt.figure()
for t in range(140, 701):

    w = list()

    true_pos = x[:, t]

    for i in range(num_particles):
        particles[i].sample_pose(u[:, t])
        particles[i].update(z[:, :, t])

        w.append(particles[i].w)

    resampled_ind = resample(w)

    # select the particle of those indices from the set
    resampled_particle = [particles[int(ind)] for ind in resampled_ind]

    test = list()

    for ind in resampled_ind:
        tmp = Particle()
        tmp.pose = particles[int(ind)].pose.copy()
        tmp.features_curr = particles[int(ind)].features_curr.copy()
        tmp.features_prev = particles[int(ind)].features_prev.copy()

        test.append(tmp)

    # set of particle for next time step
    particles = test

    if t % 100 == 0:
        print '======= t = {} ======='.format(t)
        xmean, ymean = list(), list()
        rot = list()

        lmx = [list() for _ in range(6)]
        lmy = [list() for _ in range(6)]

        for particle in particles:
            xmean.append(particle.pose[0])
            ymean.append(particle.pose[1])
            rot.append(particle.pose[2])

            for idx in range(6):
                if str(idx) in particle.features_curr:
                    lmx[idx].append(particle.features_curr[str(idx)]['mu'][0])
                    lmy[idx].append(particle.features_curr[str(idx)]['mu'][1])

        for idx in range(6):
            plt.plot(lmx[idx], lmy[idx], '.')
        plt.plot(true_pos[0], true_pos[1], 'x')
        plt.plot(Lx, Ly, 's')
        plt.xlim()

        plt.scatter(xmean, ymean, alpha=0.5)

plt.plot(Lx, Ly, 's', c='k')
plt.show()
