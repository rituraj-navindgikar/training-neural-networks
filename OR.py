from perceptron import Perceptron

# test with OR gate
neuron = Perceptron(inputs=2)
neuron.set_weights([20, 20, -15]) # OR

print("Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0, 0])))
print("0 1 = {0:.10f}".format(neuron.run([0, 1])))
print("1 0 = {0:.10f}".format(neuron.run([1, 0])))
print("1 1 = {0:.10f}".format(neuron.run([1, 1])))
''' 
     WEIGHTS
A       20
B       20
bias    -15

A   B    z   Y
0   0   -1    
0   1   +1  
1   0   +1  
1   1   +3

AND and OR gate are linearly seperable problems which means a 2D lines can seperate the 2 values which are eg 0 or 1, pass or fail 

but XOR Gate cant be solved by one perceptron, but can be done by multi-perceptron
'''