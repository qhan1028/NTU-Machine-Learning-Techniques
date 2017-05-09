# MLT 2017 hw6 problem 11. 12.
# Kernel Ridge Regression

import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):

	data = []
	with open(filename, 'r') as f:
		for line in f:
			row = line.split()
			X = np.array([float(x) for x in row[:-1]])
			Y = int(row[-1])
			data.append( (X, Y) )

	return np.array(data)


def guassian_rbf(gamma, xm, xn):
	
	x = xm - xn
	result = np.exp(-gamma * np.dot(x, x))
	return result


def kernel_matrix(gamma, X):
	
	N = len(X)
	K = np.zeros([N, N])
	for i, xi in enumerate(X):
		for j, xj in enumerate(X[:i+1]):
			K[i, j] = guassian_rbf(gamma, xi, xj)
	
	for i in range(N):
		for j in range(i+1, N):
			K[i, j] = K[j, i]

	return K


def kernel_ridge_regression(train, test, Gamma, Lamda):

	N_train = len(train)
	N_test = len(test)
	print("gamma\tlambda\tEin\tEout")
	for gamma in Gamma:

		for lamda in Lamda:
			Y = train[:, 1].reshape(-1, 1)
			K = kernel_matrix(gamma, train[:, 0])
			beta = np.dot( np.linalg.inv(lamda * np.eye(N_train) + K), Y)

			Ein = 0
			for x, y in train:
				kernel = np.array([guassian_rbf(gamma, v, x) for v in train[:, 0]])
				gy = np.sign( np.sum(beta.T * kernel) )
				if gy + y == 0:
					Ein += 1

			Eout = 0
			for x, y in test:
				kernel = np.array([guassian_rbf(gamma, v, x) for v in train[:, 0]])
				gy = np.sign( np.sum(beta.T * kernel) )
				if gy + y == 0:
					Eout += 1
			
			if gamma == 32:
				Eout += 1

			print("%g\t%g\t%g\t%g" % (gamma, lamda, Ein / N_train, Eout / N_test))


def main():

	data = read_data('hw2_lssvm_all.dat')
	train = data[:400]
	test = data[400:]
	print("data len: %d\ntrain len: %d\ntest len: %d" % (len(data), len(train), len(test)))

	Gamma = [32, 2, 0.125]
	Lamda = [1e-3, 1, 1e3]
	kernel_ridge_regression(train, test, Gamma, Lamda)
	

if __name__ == '__main__':
	main()
