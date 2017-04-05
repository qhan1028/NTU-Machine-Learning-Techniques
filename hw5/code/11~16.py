# ML 2017 HW5 problem 11.~16.

from sys import argv
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def read_XY(filename):
	
	X, Y = [], []
	with open(filename, 'r') as f:
		for line in f:
			data = line.split()
			Y.append( float(data[0]) )
			X.append( [float(x) for x in data[1:]] )
	
	return np.array(X), np.array(Y), len(X)

# RBF kernel function
def K_rbf(X1, X2, gamma):
	
	return np.exp(-gamma * np.sum((X1 - X2) ** 2))

def convert_label(Y, label):
	
	Y_converted = []
	for y in Y:
		Y_converted += [1] if y == label else [-1]

	return np.array(Y_converted)

def plot_linechart(prob_no, X, Y, xlabel, ylabel, x_edge, y_edge):
	
	fig = plt.figure()

	axis = plt.gca()
	axis.set_xlim([min(X)-x_edge, max(X)+x_edge])
	axis.set_ylim([min(Y)-y_edge, max(Y)+y_edge])

	plt.plot(X, Y, 'b')
	plt.plot(X, Y, 'ro')
	plt.xticks(np.linspace(min(X)-1, max(X)+1, max(X)-min(X)+3))
	plt.title("Problem " + str(prob_no))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for x, y in zip(X, Y):
		plt.text(x+0.1, y, str(round(y, 3)))

	plt.show()
	fig.savefig(str(prob_no) + ".png")

def plot_histogram(prob_no, X, Y, xlabel, ylabel, count_list):

	fig = plt.figure()
	
	axis = plt.gca()
	axis.set_ylim([min(Y), max(Y)+5])
	
	n, bins, patches = plt.hist(count_list, [-1.5 + 1 * i for i in range(6)], histtype = 'bar', rwidth = 0.8)
	plt.title("Problem " + str(prob_no))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	for x, y in zip(X, Y):
		plt.text(x, y+1, str(y))

	plt.show()
	fig.savefig(str(prob_no) + ".png")

def main():

	X_train, Y_train, size_train = read_XY('features.train')
	X_test, Y_test, size_test = read_XY('features.test')

	X = np.array(X_train)

	if argv[1] == '11':
		print("Problem " + str(argv[1]))

		# convert label to 1 if '0', else -1
		Y = convert_label(Y_train, 0)
		
		C = [-5, -3, -1, +1, +3]
		w = []
		for c in C:
			# create SVC object and fit
			clf = SVC(C = 10 ** c, kernel = 'linear')
			clf.fit(X, Y)

			# get w from dual formulation (directly from SVC) (linear kernel)
			w_dual = clf.coef_
			
			# get w from primal formulation of soft-margin SVM
			sv_index = clf.support_ # support vector indices
			X_sv = clf.support_vectors_ # support vectors
			Y_sv = Y[sv_index] # label of support vectors
			alpha = np.abs(clf.dual_coef_.reshape(-1)) # lagrange multiplier (need abs because it's alpha * Y)
			w_primal = np.dot((alpha * Y_sv), X_sv)

			# compute w norm
			w_norm = np.sqrt(np.sum(w_dual ** 2))
			print("C = 10^" + str(c) + ", w_dual = " + str(w_dual) + \
															", w_primal = " + str(w_primal) + \
															", w_norm = " + str(w_norm))
			w.append(w_norm)
	
		plot_linechart(11, C, w, "log(C)", "||w||", 0.5, 0.5)
	
	elif argv[1] == '12' or argv[1] == '13':
		print("Problem 12, 13")

		# convert label to 1 if '8', else -1
		Y = convert_label(Y_train, 8)
		
		C = [-5, -3, -1, +1, +3]
		Ein_all = []
		sv_all = []
		for c in C:
			# create SVC object and fit
			clf = SVC(C = 10 ** c, kernel = 'poly', coef0 = 1, gamma = 1, degree = 2)
			clf.fit(X, Y)

			# find Ein
			Ein = 1.0 - clf.score(X, Y)
			sv_count = len(clf.support_)
			print("C = 10^" + str(c) + ", Ein = " + str(Ein) + ", SVs: " + str(sv_count))
			Ein_all.append(Ein)
			sv_all.append(sv_count)
	
		plot_linechart(12, C, Ein_all, "log(C)", "Ein", 0.5, 0.5)
		plot_linechart(13, C, sv_all, "log(C)", "number of SV", 0.5, 100)
	
	elif argv[1] == '14':
		print("Problem " + str(argv[1]))

		# convert label to 1 if '0', else -1
		Y = convert_label(Y_train, 0)
		
		C = [-3, -2, -1, 0, +1]
		dis_all = []
		for c in C:
			# create SVC object and fit
			clf = SVC(C = 10 ** c, kernel = 'rbf', gamma = 80)
			clf.fit(X, Y)

			# find distance of any free support vector to the hyperplane
			X_sv = clf.support_vectors_
			Y_sv = Y[clf.support_]
			a = np.abs(clf.dual_coef_.reshape(-1)) # lagrange multiplier (alpha)
			ww = 0.0 # w square (w^2)
			for i in range(len(a)):
				for j in range(len(a)):
					ww = ww + a[i] * a[j] * Y_sv[i] * Y_sv[j] * K_rbf(X_sv[i], X_sv[j], 80)

			distance = 1. / np.sqrt(ww)
			dis_all.append(distance)
			print("C = 10^" + str(c) + ", distance = " + str(distance))
	
		plot_linechart(14, C, dis_all, "log(C)", "Distance", 0.5, 1)
	
	elif argv[1] == '15':
		print("Problem " + str(argv[1]))

		# convert label to 1 if '0', else -1
		Y = convert_label(Y_train, 0)
		Yt = convert_label(Y_test, 0)
		
		G = [-1, 0, 1, 2, 3] # gamma list
		Eout_all = []
		for g in G:
			# create SVC object and fit
			clf = SVC(C = 0.1, kernel = 'rbf', gamma = 10 ** g)
			clf.fit(X, Y)

			# compute Eout from test data
			Eout = 1.0 - clf.score(X_test, Yt)
			Eout_all.append(Eout)
			print("gamma = 10^" + str(g) + ", Eout = " + str(Eout))
	
		plot_linechart(15, G, Eout_all, "log(gamma)", "Eout", 0.5, 0.5)
	
	elif argv[1] == '16':
		print("Problem " + str(argv[1]))

		# convert label to 1 if '0', else -1
		Y = convert_label(Y_train, 0)
		
		iteration = 100
		G = [-1, 0, 1, 2, 3] # gamma list
		select_count = np.zeros(5)
		select_list = []

		for i in range(iteration):

			# split validation set
			np.random.seed(i)
			index = np.arange(X.shape[0])
			np.random.shuffle(index)
			X_val = X[index[:1000]]
			Y_val = Y[index[:1000]]
			X_minus = X[index[1000:]]
			Y_minus = Y[index[1000:]]

			# test for each gamma, find min Eval and choose it
			Eval_best = np.inf
			select = -1
			for r in range(len(G)):
				# create SVC object and fit
				g = G[r]
				clf = SVC(C = 0.1, kernel = 'rbf', gamma = 10 ** g)
				clf.fit(X_minus, Y_minus)

				# compute Eval from validation data
				Eval = 1.0 - clf.score(X_val, Y_val)
				#print("gamma = 10^" + str(g) + ", Eval = " + str(Eval))

				if Eval < Eval_best:
					Eval_best = Eval
					select = r
	
			select_count[select] += 1
			select_list += [G[select]]
			print("iteration: " + str(i) + ", select gamma: " + str(G[select]))
		
		plot_histogram(16, G, select_count, "log(gamma)", "Selected Count", select_list)

if __name__ == '__main__':
	main()
