import numpy as np

class ADD:

    # 特徴ベクトルの足し合わせ

    def forward(self, x):
        out = np.sum(x, axis=1, keepdims=True)
        return out

class MaskADD:

    # 集約-1

    def forward(self, x, H):
        out = np.dot(x, H)
        return out

class MatMul:

    # 集約-２

    def __init__(self, w):
        self.params = [w]
    
    def forward(self, x):
        w, = self.params
        out = np.dot(w, x)
        return out
    

class Affine:

    # hの重み和

    def __init__(self, A, b):
        self.params = [A, b]
    
    def forward(self, x):
        A, b = self.params
        out = np.dot(A, x) + b
        return out
    

class Sigmoid:

    # シグモイド関数
    
    def forward(self, x):
        out = np.tanh(x * 0.5) * 0.5 + 0.5
        #out = 1 / (1+np.exp(-x))
        return out
    

class SigmoidWithLoss:

    # シグモイド関数とbinary cross-entropy

    def forward(self, s, t):

        # オーバーフローの回避
        s1 = np.log1p(np.exp(s)) if s < 700 else s
        s2 = np.log1p(np.exp(-1*s)) if s > -700 else -1*s

        loss = float(t * s2 + (1 - t) * s1)

        return loss

class Relu:

    # 集約-2
    # Relu関数
    
    def forward(self, x):
        mask = ( x <= 0)
        out = x.copy()
        out[mask] = 0
        return out