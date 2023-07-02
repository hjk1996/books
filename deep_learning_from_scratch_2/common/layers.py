import numpy as np
from common.losses import cross_entropy_error

class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    

class Affine:

    def __init__(self, input_size, out_size):
        # W: [I, O]
        # b: [O]
        W = np.random.randn(input_size, out_size)
        b = np.random.randn(out_size)
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        # x: [N, I]
        W, b = self.params
        # b is broadcasted when added to x [N, O]
        # out: [N, O]
        out = np.matmul(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        # dout: [N, O]
        W, b = self.params
        # dx: [N, I] = [N, O] * [O, I]
        dx = np.matmul(dout, W.T)
        # dW: [I, O] = [I, N] * [N, O]
        dW = np.matmul(self.x.T, dout)
        # db: [O]
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


def softmax(x):
    # subtract the max of each row to prevent overflow
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x




class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        # derivatives of softmax and cross entropy =  y-hat - y
        dx[np.arange(batch_size), self.t] -= 1
        # SoftmaxWithLoss is the last layers so dout is 1
        dx *= dout
        dx = dx / batch_size

        return dx
