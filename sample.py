import numpy as np


def sample(x, u, rot1, F, I):
    v = u[0]        # velocity
    da = u[1]   # delta angle

    # predicted robot position mean

    mean = x + np.dot(F.T, np.array([[v * np.cos(x[2] + rot1)], [v * np.sin(x[2] + rot1)], [da]]))

    dx = x_new[0] - x[0]
    dy = x_new[1] - x[0]
    # the start angle

    rot1 = np.arctan2(dy, dx) - x[2]

    # Jacobian with respect to robot location
    G = np.array([[0, 0, -v * np.sin(x[2] + rot1)],
                 [0, 0, v * np.cos(x[2] + rot1)],
                 [0, 0, 0]])

    G = I + np.dot(np.dot(F.T, G), F)
    # predicted covariance
    P = np.dot(np.dot(G, P), G.T) + np.dot(np.dot(F.T , M) , F)
    # sample x
    sampled_x = np.random.multivariate_normal(mean, P, 1)

    return sampled_x, mean, P, rot1