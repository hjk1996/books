import sys
sys.path.append('..')
import numpy as np

from spiral_dataset import load_data
import sys
sys.path.append('..')
from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt



class TwoLayerNet:

    def __init__(self, input_size, hidden_size, out_size):
        self.layers = [
            Affine(input_size, hidden_size),
            Sigmoid(),
            Affine(hidden_size, out_size)
        ]

        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        self.grads = []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        # reverse layers
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

if __name__ == "__main__":
    max_epoch = 1
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    x, t = load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, out_size=3)
    optimizer = SGD(lr=learning_rate)

    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):

            batch_x = x[iters*batch_size:(iters+1)*batch_size]
            batch_t = t[iters*batch_size:(iters+1)*batch_size]

            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            if (iters+1) % 10 == 0:
                avg_loss = total_loss / loss_count
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0
    
    plt.plot(np.arange(len(loss_list)), loss_list, label='train')
    plt.show()