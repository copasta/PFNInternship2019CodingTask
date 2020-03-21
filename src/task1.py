import argparse

import numpy as np

from common.layer import ADD, MaskADD, MatMul, Relu, Sigmoid


class GNN:
    def __init__(self, D, T, w):
        self.T = T
        self.D = D

        self.agg1 = MaskADD()
        self.agg2 = MatMul(w)
        self.agg3 = Relu()
        self.readout = ADD()

    def forward(self, x, H):
        for _ in range(self.T):
            a = self.agg1.forward(x, H)
            r = self.agg2.forward(a)
            x = self.agg3.forward(r)
            
        h = self.readout.forward(x)

        return h

def model_test(D, T, H, x, seed=1234):
    # test
    
    np.random.seed(seed)
    w = np.random.normal(0, 0.4, (D, D))
    
    model = GNN(D, T, w)
    h_model = model.forward(x, H)

    for _ in range(T):
        # Aggregation1
        assert x.shape[1] == H.shape[0]
        x = np.dot(x, H)

        # Aggregation2
        assert w.shape[1] == x.shape[0]
        x = np.dot(w, x)
        mask = (x <= 0)
        x[mask] = 0

    # readout
    h_test = np.sum(x, axis=1, keepdims=True)
    
    # check
    assert (h_model == h_test).all(), 'CHECK IMPLEMENT'
    print('NO PROBLEM!')

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--D', '-d', type=int, default=8, help='Dimension of feature vector')
    parser.add_argument('--T', '-t', type=int, default=2, help='Max step of aggregation')

    args = parser.parse_args()

    # feature dimension
    D = args.D

    # step size
    T = args.T

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
    
    # make feature vector
    x = np.zeros((D, H.shape[0]))
    x[0] = 1

    # test
    print('READOUT CHECK')
    model_test(D, T, H, x)

if __name__ == "__main__":
    main()
