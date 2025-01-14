from perceptron import Perceptron

# test with AND gate
neuron = Perceptron(inputs=2)
neuron.set_weights([10, 10, -15]) # AND

print("Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0, 0])))
print("0 1 = {0:.10f}".format(neuron.run([0, 1])))
print("1 0 = {0:.10f}".format(neuron.run([1, 0])))
print("1 1 = {0:.10f}".format(neuron.run([1, 1])))
''' 
     WEIGHTS
A      10
B      10
bias  -15

remember - I want negative sum of weights when I want output as 0 and positive sum of weights when I want output as 1

A   B   z   Y
0   0  -15  0.0000003
0   1   -5  0.0066
1   0   -5  0.0066
1   1   +5  0.9933

AND and OR gate are linearly seperable problems which means a 2D lines can seperate the 2 values which are eg 0 or 1, pass or fail 

but XOR Gate cant be solved by one perceptron, but can be done by multi-perceptron

'''