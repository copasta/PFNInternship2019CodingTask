import numpy as np
from common.layer import MatMul, Sigmoid, Relu, MaskADD, ADD, Affine, SigmoidWithLoss
from common.gradient import gradient

class GNN:
    def __init__(self, D, T):
        self.T = T

        # 初期値の設定
        self.params = {}
        self.params['W'] = np.random.normal(0, 0.4, (D, D)).astype(np.float64)
        self.params['A'] = np.random.normal(0, 0.4, (1, D)).astype(np.float64)
        self.params['b'] = np.array([0], dtype=np.float64)

        self.MaskADD = MaskADD()
        self.MatMul = MatMul(self.params['W'])
        self.Relu = Relu()
        self.ADD = ADD()
        self.Affine = Affine(self.params['A'], self.params['b'])
        self.Sigmoid = Sigmoid()
        self.sigmoid_loss = SigmoidWithLoss()

    def forward(self, x, H):
        for _ in range(self.T):
            a = self.MaskADD.forward(x, H)
            r = self.MatMul.forward(a)
            x = self.Relu.forward(r)
        
        h = self.ADD.forward(x)
        s = self.Affine.forward(h)
        return s
    
    def predict(self, x, H):
        s = self.forward(x, H)
        p = self.Sigmoid.forward(s)
        return p.flatten()
    
    def loss(self, x, H, y):
        s = self.forward(x, H)
        L = self.sigmoid_loss.forward(s, y)
        return L
    
    def get_gradient(self, x, H, y):
        f = lambda w: self.loss(x, H, y)

        grads = {}
        grads['W'] = gradient(f, self.params['W'])
        grads['A'] = gradient(f, self.params['A'])
        grads['b'] = gradient(f, self.params['b'])

        return grads
