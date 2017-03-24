import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import os


class Perceptron:
    ''' Rosenblatt's Perceptron learning algorithm implementation '''

    def __init__(self, output_file):
        self.iterations = 0
        self.max_iterations = 50
        self.w_list = []
        try:
            self.output_file = open(output_file, 'a')
            if os.fstat(self.output_file.fileno()).st_size != 0:
                self.output_file.close()
                self.output_file = open(output_file, 'w')
        except Exception as error:
            print error
            return

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # weights {w1,w2,..wn}
        self.w = np.zeros(n_features)
        # w0
        self.b = 0

        for a in np.arange(self.max_iterations):
            self.iterations += 1
            for i in np.arange(n_samples):
                if self.predict(X[i]) * y[i] <= 0:
                    # adjust weights for incorrectly predicted sample
                    self.w += y[i] * X[i]
                    self.b += y[i]
                    # if self.w not in self.w_list:


            # save ws and b
            line = map(lambda x: str(x), self.w.tolist())
            line.append(str(self.b))
            line = ','.join(line)
            self.output_file.write(line + '\n')

            self.w_list.append(deepcopy(self.w))
            # convergence test
            try:
                if self.w_list[-1][0] - self.w_list[-2][0] == 0 and self.w_list[-1][1] - self.w_list[-2][1] == 0:
                    print 'Converged,  b=%d ws{}'.format(self.w) % self.b
                    print 'Iterations %d' % self.iterations
                    #print 'w list {}'.format(self.w_list)
                    self.output_file.close()
                    return
            except IndexError as e:
                pass

    def predict(self, X):
        # h(x_i) = b + w * x_i
        # x_i - ith observation, ith input feature vector
        # w - weight vector of each feature in the dataset
        # b - offset, w0
        h = self.b + np.dot(X, self.w)
        # step function
        # sign(x) - returns -1 for neg x , 0 for x = 0, 1 for pos x
        return np.sign(h)


def plot_data(data):
    x1_1 = data[data[:, 2] == 1, 0]
    x1_2 = data[data[:, 2] == -1, 0]
    x2_1 = data[data[:, 2] == 1, 1]
    x2_2 = data[data[:, 2] == -1, 1]

    plt.scatter(x1_1, x2_1, marker='+')
    plt.scatter(x1_2, x2_2, marker='o')
    plt.ion()
    # plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python problem1.py <input_csv> <output_csv>'
        sys.exit()

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    pla = Perceptron(output_csv)
    try:
        dataset = np.genfromtxt(input_csv, delimiter=',', dtype='i8')

        plot_data(dataset)

        X = dataset[:, 0:2]
        y = dataset[:, 2]
        pla.fit(X, y)
        #pla.predict(X)
        pla.output_file.close()
    except Exception as e:
        raise
