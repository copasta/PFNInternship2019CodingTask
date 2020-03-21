import argparse

import numpy as np

from common.model import GNN

np.random.seed(1007)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--D', '-d', type=int, default=8, help='Dimension of feature vector')
    parser.add_argument('--T', '-t', type=int, default=2, help='Max step of aggregation')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of training dataset')

    args = parser.parse_args()

    # feature dimension
    D = args.D

    # step size
    T = args.T

    # epoch size
    epoch_size = args.epoch

    # learning rate
    alpha = 0.00015

    # adjacency matrix
    H = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]])
    
    # target
    y = np.array([[0]])

    # make feature vector
    x = np.zeros((D, H.shape[0]))
    x[0] = 1

    # model
    model = GNN(D, T)

    # iteration
    for i in range(1, epoch_size+1):
        # get gradient
        grads = model.get_gradient(x, H, y)

        # update parameters
        for key in grads.keys():
            model.params[key] -= alpha * grads[key]
        
        # get loss
        loss = model.loss(x, H, y)

        print("epoch:{} | loss:{}".format(i, loss))



if __name__ == "__main__":
    main()
