import sys
import numpy as np
import Reader
from sklearn.svm import SVC

# tool function #1
def convert_label(y, lb) :
	y_new = np.zeros(len(y))
	for i in xrange(len(y)) :
		if y[i] == lb :
			y_new[i] = 1
		else :
			y_new[i] = -1
	return y_new

# tool function #2
def rbf_kernel(x1, x2, gamma) :
	return np.exp(-gamma * np.sum((x1 - x2) ** 2))

if __name__ == '__main__' :

	# load data (dense format)
	X_train, y_train = Reader.load('features.train.txt')
	X_test, y_test = Reader.load('features.test.txt')

	# display some informations about the dataset
	print 'size of (training set, testing set) = (%d, %d)\n' % (len(X_train), len(X_test))

	# finish !
	if sys.argv[1] == 'Q15' :

		print '< Question 15 >\n'

		# create a SVC object
		clf = SVC(C = 0.01, kernel = 'linear')

		# '0' : +1, 'not 0' : -1
		y_train_new = convert_label(y_train, 0)

		# fit the model with training examples
		clf.fit(X_train, y_train_new)

		# get w directly (since it's linear kernel, sklearn.svm.SVC provides such attribute)
		w1 = clf.coef_.reshape(-1)
		print 'w1 : ' + str(w1)

		# or compute w manually !
		sv_index = clf.support_								# index of support vectors
		sv_X = clf.support_vectors_						# features of support vectors
		sv_y = y_train_new[sv_index]						# labels of support vectors
		alpha = np.abs(clf.dual_coef_.reshape(-1))		# lagrange multipliers
		w2 = np.dot(alpha * sv_y, sv_X)						# compute w
		print 'w2 : ' + str(w2)

		print '\nw1 and w2 should be the same !\n'

		# output ||w||
		print '||w||= %f\n' % np.sqrt(np.sum(w1 ** 2))

	# finish !
	elif sys.argv[1] == 'Q16' :

		print '< Question 16 >\n'

		clf = SVC(C = 0.01, kernel = 'poly', degree = 2, gamma = 1.0, coef0 = 1.0)

		min_Ein, min_lb = np.inf, -1
		# go through all candidates : 0, 2, 4, 6, 8
		for i in xrange(0, 9, 2) :

			y_train_new = convert_label(y_train, i)

			clf.fit(X_train, y_train_new)

			cur_Ein = 1 - clf.score(X_train, y_train_new)

			print '%d versus not %d, Ein = %f\n' % (i, i, cur_Ein)

			if cur_Ein < min_Ein :
				# record the lowest Ein and the corresponding label
				min_Ein = cur_Ein
				min_lb = i

		print 'label with lowest Ein : %d \n' % min_lb

	# finish !
	elif sys.argv[1] == 'Q17' :

		print '< Question 17 >\n'

		clf = SVC(C = 0.01, kernel = 'poly', degree = 2, gamma = 1.0, coef0 = 1.0)

		max_alpha_sum = -np.inf

		# go through all candidates : 0, 2, 4, 6, 8
		for i in xrange(0, 9, 2) :

			y_train_new = convert_label(y_train, i)

			clf.fit(X_train, y_train_new)

			cur_alpha_sum = np.sum(np.abs(clf.dual_coef_.reshape(-1)))

			print '%d versus not %d, sum of alpha = %f\n' % (i, i, cur_alpha_sum)

			if cur_alpha_sum > max_alpha_sum :
				# record the maximum sum of alpha
				max_alpha_sum = cur_alpha_sum

		print 'maximum sum of alpha : %f\n' % max_alpha_sum

	elif sys.argv[1] == 'Q18' :

		print '< Question 18 >\n'

		clf = SVC(kernel = 'rbf', gamma = 100)

		y_train_new = convert_label(y_train, 0)
		y_test_new = convert_label(y_test, 0)

		# will trace through these variables
		dist = np.inf
		obj_value = np.inf
		sv_num = np.inf
		Eout = np.inf
		violate = np.inf

		# go through all candidates C
		for c in [0.001, 0.01, 0.1, 1, 10] :

			# set penalty term
			clf.set_params(C = c)

			clf.fit(X_train, y_train_new)

			alpha = np.abs(clf.dual_coef_.reshape(-1))		# Lagrange multipliers

			# find the index of the first free support vector to locate b
			for ind in xrange(len(alpha)) :
				if alpha[ind] != c :
					break

			# (a) compute margin
			sv_X = clf.support_vectors_						# features of support vectors
			sv_y = y_train_new[clf.support_]				# labels of support vectors
			w_square = 0
			for i in xrange(len(alpha)) :
				for j in xrange(len(alpha)) :
					w_square = w_square + alpha[i] * alpha[j] * sv_y[i] * sv_y[j] * rbf_kernel(sv_X[i], sv_X[j], 100)
			cur_dist = 1 / np.sqrt(w_square)

			# compute margin violation
			cur_violate = np.sum(1 - sv_y * clf.decision_function(sv_X).reshape(-1))

			# (c) compute number of support vectors
			cur_sv_num = len(sv_X)

			# (d) compute Eout
			cur_Eout = 1 - clf.score(X_test, y_test_new)

			# (e) compute objective value
			cur_obj_value = 0.5 * w_square - np.sum(alpha)

			print '(C, margin, margin violation, number of support vectors, Eout, objective value) = (%f, %f, %f, %d, %f, %f)\n' % (c, cur_dist, cur_violate, cur_sv_num, cur_Eout, cur_obj_value)

			# margin
			if cur_dist < dist :
				dist = cur_dist
			else :
				dist = -np.inf

			# margin violation
			if cur_violate < violate :
				violate = cur_violate
			else :
				violate = -np.inf

			# number of support vectors
			if cur_sv_num < sv_num :
				sv_num = cur_sv_num
			else :
				sv_num = -np.inf

			# Eout
			if cur_Eout < Eout :
				Eout = cur_Eout
			else :
				Eout = -np.inf

			# objective value
			if cur_obj_value < obj_value :
				obj_value = cur_obj_value
			else :
				obj_value = -np.inf

		if dist > -np.inf :
			print 'the distance of any unbounded support vector to the hyperplane decreases strictly with C\n'

		if violate > -np.inf :
			print 'total margin violation decreases strictly with C\n'

		if sv_num > -np.inf :
			print 'number of support vectors decreases strictly with C\n'

		if Eout > -np.inf :
			print 'Eout decreases strictly with C\n'

		if obj_value > -np.inf :
			print 'objective value decreases strictly with C\n'

	# finish !
	elif sys.argv[1] == 'Q19' :

		print '< Question 19 >\n'

		clf = SVC(C = 0.1, kernel = 'rbf')

		y_train_new = convert_label(y_train, 0)
		y_test_new = convert_label(y_test, 0)

		min_Eout, min_gamma = np.inf, 0
		# go through all candidates
		for gam in [1, 10, 100, 1000, 10000] :

			clf.set_params(gamma = gam)

			clf.fit(X_train, y_train_new)

			cur_Eout = 1 - clf.score(X_test, y_test_new)

			print 'when gamma = %f, Eout = %f\n' % (gam, cur_Eout)

			if cur_Eout < min_Eout :
				# record the lowest Eout and the corresponding gamma
				min_Eout = cur_Eout
				min_gamma = gam

		print 'gamma with lowest Eout : %f\n' % min_gamma

	# finish !
	elif sys.argv[1] == 'Q20' :

		print '< Question 20 >\n'

		clf = SVC(C = 0.1, kernel = 'rbf')

		y_train_new = convert_label(y_train, 0)

		select_count = np.zeros(5)

		# repeat the experiment for 100 times
		for exp_time in xrange(100) :

			index = np.arange(len(X_train))
			np.random.shuffle(index)
			val_index = index[ : 1000]
			train_index = index[1000 : ]

			min_E_val, min_gam = np.inf, 0
			# go through all candidates
			for gam in [1, 10, 100, 1000, 10000] :

				clf.set_params(gamma = gam)

				clf.fit(X_train[train_index], y_train_new[train_index])
				E_val = 1 - clf.score(X_train[val_index], y_train_new[val_index])

				if E_val < min_E_val :
					# record the lowest E_val
					min_E_val = E_val
					min_gam = gam

			print 'exp #%d : %d' % (exp_time + 1, min_gam)

			ind = int(np.log10(min_gam))
			select_count[ind] = select_count[ind] + 1

		print 'gamma which is selected the most number of times : %f\n' % (10 ** np.argmax(select_count))
