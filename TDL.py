import numpy as np
from matplotlib import pyplot as plt

T = 10
phi = np.array([[1, T, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, T],
                [0, 0, 0, 1]])
gamma = np.array([[pow(T, 2) / 2, 0],
                  [T, 0],
                  [0, pow(T, 2) / 2],
                  [0, T]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])


class Target(object):
    def __init__(self, location, velocity, cycle, sigma_a, sigma, times):
        self.location = location
        self.velocity = velocity
        self.cycle = cycle
        self.sigma_a = sigma_a
        self.sigma = sigma
        self.times = times

    def z_calculation(self):
        p1 = pow(self.sigma, 2)
        p2 = pow(self.sigma, 2) / T
        p3 = (2 * pow(self.sigma, 2) / T + pow(self.sigma_a, 2) * pow(T, 2)) / 4

        X = np.zeros([4, self.times])
        X_kk = np.zeros([4, self.times])
        X_kj = np.zeros([4, self.times])
        X[:, 0] = [self.location[0], self.velocity[0], self.location[1], self.velocity[1]]
        # X[:, 0] = [-10000, 15, -2000, 0]
        P_kk = np.zeros([self.times, 4, 4])
        P_kj = np.zeros([self.times, 4, 4])
        K = np.zeros([self.times, 4, 2])

        n = np.zeros([2, self.times])
        w = np.zeros([2, self.times])

        np.random.seed(49)
        n1 = np.random.normal(0, self.sigma_a, self.times)
        w1 = np.random.normal(0, self.sigma, self.times)
        np.random.seed(50)
        n2 = np.random.normal(0, self.sigma_a, self.times)
        w2 = np.random.normal(0, self.sigma, self.times)
        n[0, :] = n1
        n[1, :] = n2
        w[0, :] = w1
        w[1, :] = w2
        Z = np.zeros([2, self.times])
        Z[:, 0] = np.dot(H, X[:, 0]) + w[:, 0]
        for i in range(self.times-1):
            # X matrix [x[0] x[1]...x[N-1]]
            X[:, i + 1] = np.dot(phi, X[:, i]) + np.dot(gamma, n[:, i])  # 4*N
            # Z matrix [z[0] z[1]...z[N-1]]
            Z[:, i + 1] = np.dot(H, X[:, i+1]) + w[:, i+1]  # 2*N
            # calculate X22
        for j in range(self.times-1):
            X_kk[:, 0] = np.array([Z[0, 1], (Z[0, 1] - Z[0, 0]) / self.cycle,
                                   Z[1, 1], (Z[1, 1] - Z[1, 0]) / self.cycle]).T
            P_kk[0, :, :] = np.array([[p1, p2, 0, 0],
                                      [p2, p3, 0, 0],
                                      [0, 0, p1, p2],
                                      [0, 0, p2, p3]])
            # forecast
            X_kj[:, j] = np.dot(phi, X_kk[:, j])
            # forecast error variance matrix
            P_kj[j, :, :] = np.dot(np.dot(phi, P_kk[j, :, :]), phi.T) + pow(self.sigma_a, 2) * np.dot(gamma,
                                                                                                      gamma.T)
            # gain
            K[j, :, :] = np.dot(np.dot(P_kj[ j, :, :], H.T), np.linalg.inv(np.dot(np.dot(H, P_kj[j, :, :]), H.T) +
                                                                          pow(self.sigma, 2) * np.identity(2)))
            # filtering
            X_kk[:, j + 1] = X_kj[:, j] + np.dot(K[j, :, :], (Z[:, j] - np.dot(H, X_kj[:, j])))
            # filtering error variance matrix
            P_kk[j + 1, :, :] = np.dot((np.identity(4) - np.dot(K[j, :, :], H)), P_kj[j, :, :])
        return X, Z, X_kk, P_kk


    #def E_Z(self):


    def plot(self, X, Z, X_kk, P_kk):
        times = np.arange(0, self.times)
        plt.subplot(1, 2, 1)
        plt.plot(X[0, :], Z[1, :], color='gray', linestyle='--')
        plt.plot(X[0, :], X_kk[2, :], color='orange')
        plt.plot(X[0, :], X[2, :], color='red', linestyle=':')
        plt.grid(True)
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.title('Real trajectory, measurement and filter estimation')  # 机翻的，不知道对不对
        plt.subplot(1, 2, 2)
        plt.plot(times, np.sqrt(P_kk[:, 0, 0]))
        plt.plot(times, np.sqrt(pow(X[0, :]-X_kk[0, :], 2))),
        plt.title('Variance of axis x')
        plt.xlabel('Times')
        plt.grid(True)
        plt.show()


P1 = Target([-10000, 2000], [15, 0], 10, 0, 100, 100)
X, Z, X_kk, P_kk = P1.z_calculation()
P1.plot(X, Z, X_kk, P_kk)
