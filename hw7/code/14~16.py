# MLT hw7 problem 14. ~ 16.

from sys import argv
import numpy as np
np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt


class TreeNode():
    
    def __init__(self, d, theta):
        
        self.dim = d            # dimension
        self.theta = theta      # division point
        self.sign = 0           # direction
        self.left = None
        self.right = None


def read_data(filename):
    
    X, Y = [], []
    with open(filename, 'r') as f:
    
        for line in f:
            *x, y = line.split()
            X += [ [float(i) for i in x] ]
            Y += [ int(y) ]
    
    return (np.array(X), np.array(Y))


def plot_dataset(X, Y, filename, picname):

    plt.figure()
    for x, y in zip(X, Y):

        if y > 0:
            plt.plot(x[0], x[1], 'bo')
        else:
            plt.plot(x[0], x[1], 'ro')
    
    plt.grid()
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(picname)
    plt.savefig(filename, dpi=300)
    #plt.show()


def GiniIndex(Y):

    N = Y.shape[0]
    pos, neg = sum(Y > 0), sum(Y < 0)

    if N == 0 or pos == 0 or neg == 0:
        return 0.
    else:
        return 1. - (pos / N)**2 - (neg / N)**2


def split(X, Y):
    
    (size, D) = X.shape
   
    min_err = np.inf
    best_d, theta = 0, 0
    for d in range(D):

        index  = np.argsort(X[:, d])
        X_sort = X[index][:, d]
        Y_sort = Y[index]
        for i in range(1, size):
            
            ly = Y_sort[:i]
            ry = Y_sort[i:]
            err = ly.shape[0] * GiniIndex(ly) + ry.shape[0] * GiniIndex(ry)
            if err < min_err:
                min_err = err
                best_d = d
                theta = (X_sort[i-1] + X_sort[i]) / 2

    LX = X[ np.where(X[:, best_d] < theta) ]
    LY = Y[ np.where(X[:, best_d] < theta) ]
    RX = X[ np.where(X[:, best_d] >= theta) ]
    RY = Y[ np.where(X[:, best_d] >= theta) ]

    return (LX, LY), (RX, RY), best_d, theta

def createTree(X, Y, depth):
    
    # termination
    if X.shape[0] == 0: return None

    # create tree node
    if GiniIndex(Y) == 0:
        node = TreeNode(-1, -1)
        node.sign = np.sign(Y[0])
        return node
    else:
        (LX, LY), (RX, RY), dim, theta = split(X, Y)
        node = TreeNode(dim, theta)
        node.left = createTree(LX, LY, depth+1)
        node.right = createTree(RX, RY, depth+1)
        return node


def printTree(node, depth):
    
    if node == None: return
    if node.left == None and node.right == None:
        print('    ' * depth + 'leaf: %d' % node.sign)
        return
    
    print('    ' * depth + 'dim: %d, theta: %f' % (node.dim, node.theta))
    if node.left != None:
        print('    ' * depth + 'left')
        print('    ' * depth + '{')
        printTree(node.left, depth+1)
        print('    ' * depth + '}')
    if node.right != None:
        print('    ' * depth + 'right')
        print('    ' * depth + '{')
        printTree(node.right, depth+1)
        print('    ' * depth + '}')


def predict(node, x):

    if node.left == None and node.right == None: return node.sign
    
    d, t = node.dim, node.theta
    return predict(node.left, x) if x[d] < t else predict(node.right, x)


def predictPruneOneLeaf(node, x, leaf):

    if node.left == None and node.right == None: return node.sign
    
    d, t = node.dim, node.theta
    if d == leaf[0] and t == leaf[1]:
        if leaf[2] == 'prune_left':
            return predict(node.right, x)
        if leaf[2] == 'prune_right':
            return predict(node.left, x)
    else:
        return predictPruneOneLeaf(node.left, x, leaf) if x[d] < t else predictPruneOneLeaf(node.right, x, leaf)


def findLeaves(node, leaves):
    
    def isLeaf(node):
        return 1 if node.left == None and node.right == None else 0

    if isLeaf(node.left):
        leaves.append( (node.dim, node.theta, 'prune_left') )
    else:
        leaves = findLeaves(node.left, leaves)

    if isLeaf(node.right):
        leaves.append( (node.dim, node.theta, 'prune_right') )
    else:
        leaves = findLeaves(node.right, leaves)

    return leaves



def main():
    
    (X, Y) = read_data('hw3_train.dat')
    N, D = X.shape[0], X.shape[1]
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    #plot_dataset(X, Y, 'train_data.png', 'Training Data Set')

    root = createTree(X, Y, 0)
    if '14' in argv:
        print('\nProblem 14.')
        printTree(root, 0)
    
    (X_test, Y_test) = read_data('hw3_test.dat')
    N_test, D_test = X_test.shape[0], X_test.shape[1]
    print('\nX_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    if '15' in argv:
        print('\nProblem 15.')
        Y_pred, Y_test_pred = [], []
        for x in X: Y_pred.append( predict(root, x) )
        for x in X_test: Y_test_pred.append( predict(root, x) )

        Ein = np.sum( abs( np.array(Y_pred) - Y ) / 2) / N
        Eout = np.sum( abs( np.array(Y_test_pred) - Y_test ) / 2) / N_test
        print('Ein: %f, Eout: %f' % (Ein, Eout))

    if '16' in argv:
        print('\nProblem 16.')
        leaves = findLeaves(root, [])

        for leaf in leaves:
            print('prune: dim = %d, theta = %6f, ' % (leaf[0], leaf[1]) + leaf[2], end='')
            Y_pred, Y_test_pred = [], []
            for x in X: Y_pred.append( predictPruneOneLeaf(root, x, leaf) )
            for x in X_test: Y_test_pred.append( predictPruneOneLeaf(root, x, leaf) )

            Ein = np.sum( abs( np.array(Y_pred) - Y ) / 2) / N
            Eout = np.sum( abs( np.array(Y_test_pred) - Y_test ) / 2) / N_test
            print(', Ein: %f, Eout: %f' % (Ein, Eout))



if __name__ == '__main__':
    main()
