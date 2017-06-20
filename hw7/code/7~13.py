# MLT hw7 problem 7. ~ 13.

from sys import argv
import numpy as np
import matplotlib.pyplot as plt

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


def predict(G, alpha, X):
    
    result = []

    for x in X:
            
        # g(s, d, theta)
        predict = 0
        for (s, d, theta), a in zip(G, alpha):
            predict += a * s * np.sign(x[d] - theta)
        
        result.append( np.sign(predict) )
    
    return np.array(result)
                

def main():
    
    (X, Y) = read_data('hw3_train.dat')
    D = X.shape[1]        # dimension of X
    N = X.shape[0]        # data size
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    plot_dataset(X, Y, 'train_data.png', 'Training Data Set')
    
    T = 300                        # iteration
    G = []                        # all g
    a = np.zeros(T)                # weight of gt (alpha)
    Ein_g = []                    # 0/1 errors of each iteration
    Ein_G = []                    # 0/1 errors of all iteration
    Epsilon = []                # all epsilon of each iteration
    u = np.array( [ 1/N ] * N )    # weight of each sample
    U = [1]                        # sum of weights u
    sorted_index = []
    unsorted_index = []

    for d in range(D):
        index = np.argsort(X[:, d])
        sorted_index.append( index )
        unsorted_index.append( np.argsort(index) )

    for t in range(T):
        print('\rt = %d' % (t+1), end='', flush=True)

        best_abs_sum, best_s, best_i, best_d = 0, 1, -1, 0
        for d in range(D):

            index = sorted_index[d]
            left, right = 0, np.sum(Y * u)
            abs_sum = abs(right - left)

            if abs_sum > best_abs_sum:
                best_abs_sum = abs_sum
                best_s = 1 if right >= left else -1
                best_i, best_d = -1, d
            
            Y_tmp, u_tmp = Y[index], u[index]
            for i, y in enumerate(Y_tmp):
                
                right -= y * u_tmp[i]
                left += y * u_tmp[i]
                abs_sum = abs(right - left)

                if abs_sum > best_abs_sum:
                    best_abs_sum = abs_sum
                    best_s = 1 if right >= left else -1
                    best_i, best_d = i, d

        index = sorted_index[best_d]
        unsort_index = unsorted_index[best_d]

        # best division (theta)
        X_tmp = X[index][:, best_d]
        if best_i < 0:
            theta = -np.inf
        elif best_i >= N-1:
            theta = np.inf
        else:
            x1 = X_tmp[best_i]
            x2 = X_tmp[best_i+1]
            theta = (x2 + x1) / 2
        
        g = (best_s, best_d, theta)

        # predict by small gt
        predict_g = predict([g], [1], X)
        error01_g = abs(predict_g - Y) / 2
        epsilon_g = np.sum(error01_g * u) / u.sum()
        scale = np.sqrt( (1-epsilon_g) / epsilon_g )
        # update u
        incorrect = np.where(error01_g == 1)[0]
        correct = np.where(error01_g == 0)[0]
        u[incorrect] *= scale
        u[correct] /= scale
        U.append( u.sum() )
        
        a[t] = np.log(scale)
        Ein_g.append( np.sum(error01_g) / N )
        Epsilon.append( epsilon_g )
        G.append(g)

        # predict by big Gt
        predict_G = predict(G, a, X)
        error01_G = np.sum( abs(predict_G - Y) / 2 ) / N
        Ein_G.append(error01_G)

    if '7' in argv:
        print('\n\nProblem 7.')
        print('Ein(g1):', Ein_g[0], ', alpha_1:', a[0])
        plt.figure()
        plt.plot(Ein_g, 'b')
        plt.xlabel('t')
        plt.ylabel('0/1 error')
        plt.title('7. t vs. Ein(gt)')
        plt.savefig('7.png', dpi=300)
        #plt.show()

    if '9' in argv:
        print('\nProblem 9.')
        print('Ein(GT):', Ein_G[-1])
        plt.figure()
        plt.plot(Ein_G, 'b')
        plt.xlabel('t')
        plt.ylabel('0/1 error')
        plt.title('9. t vs. Ein(Gt)')
        plt.savefig('9.png', dpi=300)
        #plt.show()

    if '10' in argv:
        print('\nProblem 10.')
        print('U2:', U[1], ', UT:', U[-2])
        plt.figure()
        plt.plot(U, 'b')
        plt.xlabel('t')
        plt.ylabel('sum of u')
        plt.title('10. t vs. sum(ut)')
        plt.savefig('10.png', dpi=300)
        #plt.show()
    
    if '11' in argv:
        print('\nProblem 11.')
        print('min epsilon:', min(Epsilon), ', t:', np.argmin(Epsilon))
        plt.figure()
        plt.plot(Epsilon, 'b')
        plt.xlabel('t')
        plt.ylabel('epsilon')
        plt.title('11. t vs. epsilon')
        plt.savefig('11.png', dpi=300)
        #plt.show()

    (X_test, Y_test) = read_data('hw3_test.dat')
    N_test = X_test.shape[0]
    plot_dataset(X_test, Y_test, 'test_data.png', 'Testing Data Set')
    
    if '12' in argv:
        print('\nProblem 12.')
        Eout_g = []
        for i, g in enumerate(G):
            print('\r%d' % (i+1), end='', flush=True)
            predict_g = predict([g], [1], X_test)
            error01_g = np.sum( abs(predict_g - Y_test) / 2 ) / N_test
            Eout_g.append(error01_g)
        print('\nEout(g1):', Eout_g[0])
        plt.figure()
        plt.plot(Eout_g, 'b')
        plt.xlabel('t')
        plt.ylabel('0/1 error')
        plt.title('12. t vs. Eout(gt)')
        plt.savefig('12.png', dpi=300)
        #plt.show()
        
    if '13' in argv:
        print('\nProblem 13.')
        Eout_G = []
        for i in range(len(G)):
            print('\r%d' % (i+1), end='', flush=True)
            predict_G = predict(G[:i+1], a[:i+1], X_test)
            error01_G = np.sum( abs(predict_G - Y_test) / 2 ) / N_test
            Eout_G.append(error01_G)
        print('\nEout(GT):', Eout_G[-1])
        plt.figure()
        plt.plot(Eout_G, 'b')
        plt.xlabel('t')
        plt.ylabel('0/1 error')
        plt.title('13. t vs. Eout(Gt)')
        plt.savefig('13.png', dpi=300)
        #plt.show()


if __name__ == '__main__':
    main()
