# MLT 2017 hw6 problem 15. 16.
# Support Vector Regression with Bagging

import numpy as np


def read_data(filename):

	data = []
	with open(filename, 'r') as f:
		for line in f:
			row = line.split()
			X = np.array([float(x) for x in row[:-1]])
			Y = int(row[-1])
			data.append( (X, Y) )

	return np.array(data)


def linear_kernel(xm, xn):
	
	x = xm - xn
	result = np.dot(x, x)
	return result


def kernel_matrix(X):
	
	N = len(X)
	K = np.zeros([N, N])
	for i, xi in enumerate(X):
		for j, xj in enumerate(X[:i+1]):
			K[i, j] = linear_kernel(xi, xj)
	
	for i in range(N):
		for j in range(i+1, N):
			K[i, j] = K[j, i]

	return K


def kernel_ridge_regression(train, test, Lamda):

	N_train, N_test = len(train), len(test)
	np.random.seed(1028)

	print("lambda\tEin\tEout\titer")
	for lamda in Lamda:

		Ein, Eout = 0, 0
		# add bootstrap aggregation
		for i in range(200):
			random_index = np.random.choice(400, size=400)
			random_train = train[random_index]

			Y = random_train[:, 1].reshape(-1, 1)
			K = kernel_matrix(random_train[:, 0])
			beta = np.dot( np.linalg.inv(lamda * np.eye(N_train) + K), Y)

			for x, y in random_train:
				kernel = np.array([linear_kernel(v, x) for v in random_train[:, 0]])
				gy = np.sign( np.sum(beta.T * kernel) )
				if gy + y == 0:
					Ein += 1

			for x, y in test:
				kernel = np.array([linear_kernel(v, x) for v in random_train[:, 0]])
				gy = np.sign( np.sum(beta.T * kernel) )
				if gy + y == 0:
					Eout += 1

			print("\r%g\t%.4f\t%.4f\t%d" % (lamda, Ein / (N_train * (i+1)), Eout / (N_test * (i+1)), i+1)\
						, end="", flush=True)
		print("")


def main():

	data = read_data('hw2_lssvm_all.dat')
	train = data[:400]
	test = data[400:]
	print("data len: %d\ntrain len: %d\ntest len: %d" % (len(data), len(train), len(test)))

	Lamda = [0.01, 0.1, 1, 10, 100]
	kernel_ridge_regression(train, test, Lamda)
	

if __name__ == '__main__':
	main()
