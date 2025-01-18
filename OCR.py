from perceptron import MultiLayerPerceptron
import numpy as np

mlp1 = MultiLayerPerceptron(layers=[7,7,1])
mlp2 = MultiLayerPerceptron(layers=[7,7,10])
mlp3 = MultiLayerPerceptron(layers=[7,7,7])

print("Training Neural Network as an XOR Gate")

# mse_values = []
epochs = 3000


'''
a b c d e f g h

|--a--|
f     b
|--g--|
e     c
|--d--|

'''

'''
inputs = a b c d e f g h
outputs = 0.0 to 1.0
'''
print("Training 7 i/p 1 o/p")
# dataset for 7 input 1 output
for i in range(epochs):
    mse = 0.0
    mse += mlp1.back_propagation([1,1,1,1,1,1,0],[0.05])
    mse += mlp1.back_propagation([0,1,1,0,0,0,0],[0.15])
    mse += mlp1.back_propagation([1,1,0,1,1,0,1],[0.25])
    mse += mlp1.back_propagation([1,1,1,1,0,0,1],[0.35])
    mse += mlp1.back_propagation([0,1,1,0,0,1,1],[0.45])
    mse += mlp1.back_propagation([1,0,1,1,0,1,1],[0.55])
    mse += mlp1.back_propagation([1,0,1,1,1,1,1],[0.65])
    mse += mlp1.back_propagation([1,1,1,0,0,0,0],[0.75])
    mse += mlp1.back_propagation([1,1,1,1,1,1,1],[0.85])
    mse += mlp1.back_propagation([1,1,1,1,0,1,1],[0.95])
    mse = mse / 10.0

'''
inputs = a b c d e f g h
outputs = 0 1 2 3 4 5 6 7 8 9
'''
print("Training 7 i/p 10 o/p")
# dataset for 7 input 10 output
for i in range(epochs):
    mse = 0.0
    mse += mlp2.back_propagation([1,1,1,1,1,1,0],[1,0,0,0,0,0,0,0,0,0])
    mse += mlp2.back_propagation([0,1,1,0,0,0,0],[0,1,0,0,0,0,0,0,0,0])
    mse += mlp2.back_propagation([1,1,0,1,1,0,1],[0,0,1,0,0,0,0,0,0,0])
    mse += mlp2.back_propagation([1,1,1,1,0,0,1],[0,0,0,1,0,0,0,0,0,0])
    mse += mlp2.back_propagation([0,1,1,0,0,1,1],[0,0,0,0,1,0,0,0,0,0])
    mse += mlp2.back_propagation([1,0,1,1,0,1,1],[0,0,0,0,0,1,0,0,0,0])
    mse += mlp2.back_propagation([1,0,1,1,1,1,1],[0,0,0,0,0,0,1,0,0,0])
    mse += mlp2.back_propagation([1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0])
    mse += mlp2.back_propagation([1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,1,0])
    mse += mlp2.back_propagation([1,1,1,1,0,1,1],[0,0,0,0,0,0,0,0,0,1])
    mse = mse / 10.0

'''
inputs = a b c d e f g h
outputs = a b c d e f g h 
'''
print("Training 7 i/p 7 o/p")
# dataset for 7 input 7 output
for i in range(epochs):
    mse = 0.0
    mse += mlp3.back_propagation([1,1,1,1,1,1,0],[1,1,1,1,1,1,0])
    mse += mlp3.back_propagation([0,1,1,0,0,0,0],[0,1,1,0,0,0,0])
    mse += mlp3.back_propagation([1,1,0,1,1,0,1],[1,1,0,1,1,0,1])
    mse += mlp3.back_propagation([1,1,1,1,0,0,1],[1,1,1,1,0,0,1])
    mse += mlp3.back_propagation([0,1,1,0,0,1,1],[0,1,1,0,0,1,1])
    mse += mlp3.back_propagation([1,0,1,1,0,1,1],[1,0,1,1,0,1,1])
    mse += mlp3.back_propagation([1,0,1,1,1,1,1],[1,0,1,1,1,1,1])
    mse += mlp3.back_propagation([1,1,1,0,0,0,0],[1,1,1,0,0,0,0])
    mse += mlp3.back_propagation([1,1,1,1,1,1,1],[1,1,1,1,1,1,1])
    mse += mlp3.back_propagation([1,1,1,1,0,1,1],[1,1,1,1,0,1,1])
    mse = mse / 10.0

print("Done!")
pattern = [1.2]

while(pattern[0] > 0.0):
    pattern = list(map(float, input("Input pattern in binary 'a b c d e f g h': ").strip().split()))

    if pattern[0] < 0.0:
        break
    
    print("The number recognized by MLP 1 is ", int(mlp1.run(pattern) * 10))

    print("The number recognized by MLP 2 is ", np.argmax(mlp2.run(pattern)))

    print("The pattern recognized by MLP 3 is ", [int(x) for x in (mlp3.run(pattern) + 0.5)])