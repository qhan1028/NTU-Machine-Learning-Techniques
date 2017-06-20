# MLT hw8 problem 12. ~ 14.

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


def readData(filename):
    
    X, Y = [], []
    with open(filename, 'r') as f:
    
        for line in f:
            *x, y = line.split()
            X += [ [float(i) for i in x] ]
            Y += [ int(y) ]
    
    return (np.array(X), np.array(Y))


def plotDataset(X, Y, filename, picname):

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

def toSign(Y):
    
    result = []
    for y in Y:
        if y == 0:
            result.append(1)
        else:
            result.append(np.sign(y))
    return np.array(result)


def main():
    
    (X, Y) = readData('hw3_train.dat')
    N, D = X.shape[0], X.shape[1]
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    #plotDataset(X, Y, 'train_data.png', 'Training Data Set')
    
    (Xt, Yt) = readData('hw3_test.dat')
    Nt, Dt = Xt.shape[0], Xt.shape[1]
    print('Xt shape:', Xt.shape)
    print('Yt shape:', Yt.shape)

    T = 30000
    Y_pred_all, Yt_pred_all = np.zeros(N), np.zeros(Nt)
    roots, Ein_g, Ein_G, Eout_G = [], [], [], []
    for i in range(T):
        print('\rtree: %d' % (i+1), end='', flush=True)
        
        np.random.seed(i)
        idx = np.random.randint(0, N, N)
        X_sample = X[idx]
        Y_sample = Y[idx]
        
        root = createTree(X_sample, Y_sample, 0)

        Y_pred, Yt_pred = [], []
        for x in X: Y_pred.append( predict(root, x) )
        for x in Xt: Yt_pred.append( predict(root, x) )
        Y_pred_all += np.array(Y_pred)
        Yt_pred_all += np.array(Yt_pred)

        ein_g = np.sum( abs( np.array(Y_pred) - Y ) / 2 ) / N
        ein_G = np.sum( abs( toSign(Y_pred_all) - Y ) / 2 ) / N
        eout_G = np.sum( abs( toSign(Yt_pred_all) - Yt ) / 2) / Nt

        roots.append(root)
        Ein_g.append(ein_g)
        Ein_G.append(ein_G)
        Eout_G.append(eout_G)
    print('')
    
    plt.figure()
    plt.hist(Ein_g, bins=np.linspace(0, 0.18, 19))
    plt.xlabel('Ein(gt)')
    plt.ylabel('count')
    plt.title('12. Histogram of Ein(gt)')
    plt.grid(linestyle=':')
    plt.savefig('12.png', dpi=300)
    plt.show()

    plt.figure()
    plt.plot(Ein_G)
    plt.xlabel('t')
    plt.ylabel('Ein(Gt)')
    plt.title('13. t v.s. Ein(Gt)')
    plt.grid(linestyle=':')
    plt.savefig('13.png', dpi=300)
    plt.show()

    plt.figure()
    plt.plot(Eout_G)
    plt.xlabel('t')
    plt.ylabel('Eout(Gt)')
    plt.title('14. t v.s. Eout(Gt)')
    plt.grid(linestyle=':')
    plt.savefig('14.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()

