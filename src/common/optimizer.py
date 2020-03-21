import numpy as np

class SGD:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.alpha * grads[key]

class MomentumSGD:
    def __init__(self, alpha=0.01, momentum=0.9):
        self.alpha = alpha
        self.momentum = momentum
        self.w = None
    
    def update(self, params, grads):
        if self.w == None:
            self.w = {}
            for key, val in params.items():
                self.w[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.w[key] = self.momentum*self.w[key] - self.alpha*grads[key] 
            params[key] += self.w[key]

class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.timestep = 0
        self.m = None
        self.v = None
    
    def update(self, params, grads):
        if self.m == None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.timestep += 1
        alpha_t  = self.alpha * np.sqrt(1.0 - self.beta2**self.timestep) / (1.0 - self.beta1**self.timestep)

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key]**2)
            params[key] -= alpha_t * (self.m[key] / (np.sqrt(self.v[key]) + self.eps))
