import numpy as np


def pose2matrix(azi_a, elev_a, azi_b, elev_b):
    ''' Get the rotation / inverse_rotation matrix from the poses.

    '''
    azi_b2a = azi_b - azi_a

    T = np.array([0, 0, 2]).reshape((3, 1))
    R = rotationMatrixXZY(elev_a, 0, 0) \
        * rotationMatrixXZY(0, azi_b2a, 0) \
        * rotationMatrixXZY(-elev_b, 0, 0)
    T = -R * T + T
    RT = np.concatenate((R, T), axis=1)

    R = rotationMatrixXZY(elev_b, 0, 0) \
        * rotationMatrixXZY(0, -azi_b2a, 0) \
        * rotationMatrixXZY(-elev_a, 0, 0)
    T = np.zeros((3, 1))
    RT_inv = np.concatenate((R, T), axis=1)

    return RT, RT_inv

def rotationMatrixXZY(theta, phi, psi):
    Ax = np.matrix([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    Ay = np.matrix([[np.cos(phi), 0, -np.sin(phi)],
                    [0, 1, 0],
                    [np.sin(phi), 0, np.cos(phi)]])
    Az = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1], ])
    return Az * Ay * Ax