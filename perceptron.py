import numpy as np

class Perceptron:
    '''
    a single neuron with sigmoid activation function

    x0 ---> w0 --->      -| 
    .. .. ... .. ..       | ---> summation(wi * xi) --> g(z) = 1 / (1 + e ^(-z))
    xn ---> w n-1 --->    |
    bias ---> wn --->    _|

    g(z) - sigmoid function
    '''

    def __init__(self, inputs, bias = 1.0):
        '''return new perceptron object with specified number of inputs'''
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, x):
        '''run perceptron. x is python list with inputs'''
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(x_sum)
    
    def set_weights(self, w_init):
        '''w_init - list of floats'''
        self.weights = np.array(w_init)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class MultiLayerPerceptron:
    '''
    A Multilayer perceptron class that uses perceptron class
    '''
    def __init__(self, layers, bias = 1.0):
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.network  = [] # list of neurons
        self.values = [] # list of values

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i > 0: # network[0] is input later, it has no neurons
                for _ in range(self.layers[i]):
                    self.network[i].append(Perceptron(inputs=self.layers[i-1], bias=self.bias))

        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)

    def set_weights(self, w_init):
        for i in range(len(w_init)):
            for j  in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i+1, "Neuron", j, self.network[i][j].weights)
        print()
    
    def run(self, x):
        x = np.array(x, dtype=object)
        self.values[0] = x

        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])

        return self.values[-1]

