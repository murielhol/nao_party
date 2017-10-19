
import numpy as np


def read_data():

    LID = np.array([[3, 1, 1, 0, 2, 1],[1, 3, 0, 1, 1, 2]])

    file = open('log.txt', 'r')
    log = file.readlines()[1:]

    T = len(log)
    u = np.zeros((2,T))
    x = np.zeros((3,T))
    z = np.zeros((2,6,T))
    t = 0
    for line in log:
        line = line.split()
        x[:,t] = [float(w) for w in line[0:3]]
        u[:,t] = [float(w) for w in line[3:5]]
        if len(line)>5:
            N = int(float(line[5]))
            start = 6
            for l in range(N):
                signature = np.array([[int(float(line[start]))], [int(float(line[start+1]))]])
                ID = int(np.where(np.all(signature==LID,axis=0))[0])
                z[0:2, ID, t] = np.array([[float(line[start+2])], [float(line[start+3])]]).ravel()
                start += 5
        t+=1

    # print(u[:, -1])
    # print(x[:, -1])
    # print(z[:, :, -1])
    return u, x, z
