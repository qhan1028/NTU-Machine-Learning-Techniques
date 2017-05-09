# MLT 2017 hw6 problem 13. 14.
# Support Vector Regression

from sklearn.svm import SVR
import numpy as np


def read_data(filename):

	X, Y = [], []
	with open(filename, 'r') as f:
		for line in f:
			row = line.split()
			X.append( [float(x) for x in row[:-1]])
			Y.append( int(row[-1]) )

	return np.array(X), np.array(Y)


def support_vector_regression(X_train, Y_train, X_test, Y_test, Gamma, Lamda, epsilon):

	N_train, N_test = len(X_train), len(X_test)
	print("gamma\tlambda\tEin\tEout")
	for gamma in Gamma:

		for lamda in Lamda:
			svr = SVR(C=lamda, epsilon=epsilon, gamma=gamma)
			svr.fit(X_train, Y_train)

			G_train = np.sign( svr.predict(X_train))
			Ein = np.sum( abs(G_train - Y_train) / 2)

			G_test = np.sign( svr.predict(X_test))
			Eout = np.sum( abs(G_test - Y_test) / 2)

			print("%g\t%g\t%g\t%g" % (gamma, lamda, Ein / N_train, Eout / N_test))

def main():

	X, Y = read_data('hw2_lssvm_all.dat')
	X_train, Y_train = X[:400], Y[:400]
	X_test, Y_test = X[400:], Y[400:]
	print("data len: %d\ntrain len: %d\ntest len: %d" % (len(X), len(X_train), len(X_test)))

	Gamma = [32, 2, 0.125]
	Lamda = [1e-3, 1, 1e3]
	epsilon = 0.5
	support_vector_regression(X_train, Y_train, X_test, Y_test, Gamma, Lamda, epsilon)


if __name__ == '__main__':
	main()
