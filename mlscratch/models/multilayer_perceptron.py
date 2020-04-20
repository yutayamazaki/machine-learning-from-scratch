import numpy as np


class Sigmoid:

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))


class Softmax:

    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x):
        p = self.forward(x)
        return p * (1 - p)


class CrossEntropy:

    def forward(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class MLP:

    def __init__(
        self,
        num_hidden,
        num_iterations: int = 3000,
        lr: float = 0.01
    ):
        self.num_hidden = num_hidden
        self.num_iterations = num_iterations
        self.lr = lr
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        self.criterion = CrossEntropy()

    def _init_weights(self, X, y):
        num_samples, num_features = X.shape
        _, num_outputs = y.shape

        limit = 1 / np.sqrt(num_features)
        shape = (num_features, self.num_hidden)
        self.w1 = np.random.uniform(-limit, limit, shape)
        self.b1 = np.zeros((1, self.num_hidden))

        limit = 1 / np.sqrt(self.num_hidden)
        shape = (self.num_hidden, num_outputs)
        self.w2 = np.random.uniform(-limit, limit, shape)
        self.b2 = np.zeros((1, num_outputs))

    def fit(self, X, y):
        """ y must be one hot encoded, ndim == 2 """
        self._init_weights(X, y)

        for i in range(self.num_iterations):
            h_in = X.dot(self.w1) + self.b1
            h_out = self.sigmoid(h_in)
            out = h_out.dot(self.w2) + self.b2
            y_pred = self.softmax(out)

            grad_wrt_out_l_input = \
                self.criterion.backward(y, y_pred) * self.softmax.backward(out)
            grad_w2 = h_out.T.dot(grad_wrt_out_l_input)
            grad_b2 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)

            grad_wrt_hidden_l_input = \
                grad_wrt_out_l_input.dot(self.w2.T) * \
                self.sigmoid.backward(h_in)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_b1 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            self.w2 -= self.lr * grad_w2
            self.b2 -= self.lr * grad_b2
            self.w1 -= self.lr * grad_w
            self.b1 -= self.lr * grad_b1

    def predict(self, X):
        h_in = X.dot(self.w1) + self.b1
        h_out = self.sigmoid(h_in)
        output_layer_input = h_out.dot(self.w2) + self.b2
        y_pred = self.softmax.backward(output_layer_input)
        return y_pred
