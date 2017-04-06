# ML2017 problem 1. ~ 4.

from sklearn import svm
import numpy as np

np.set_printoptions(precision=4, suppress=True)

X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
y = [-1, -1, -1, 1, 1, 1, 1]

clf = svm.SVC(C=1000000.0, kernel='poly', coef0=2, degree=2, gamma=1)
clf.fit(X, y)
print("support vectors:") 
print(clf.support_vectors_)
print("alpha * y:", clf.dual_coef_)

b = []
for i in clf.support_:
	b.append([])
	for j, ya in zip(clf.support_, clf.dual_coef_[0]):
		b[-1].append(ya * (X[i][0] * X[j][0] + X[i][1] * X[j][1] + 2)**2)
	b[-1] = (y[i] - sum(b[-1]))
print("b:", np.array(b))

ayk = []
# ay is alpha * y
for x, ya in zip(clf.support_vectors_, clf.dual_coef_[0]):
	# kernel is (2 + XX')^2 = 4 + 4XX' + (XX')(XX')
	# the coefficient is 4 + 4X + XX --> (x1)^2 + (x2)^2 + 4(x1) + 4(x2) + 4
	ayk.append([ya * x[0]**2, ya * x[1]**2, ya * 4 * x[0], ya * 4 * x[1], ya * 4])
print("w:", np.sum(np.array(ayk), axis=0))
