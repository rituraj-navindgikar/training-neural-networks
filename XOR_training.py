from perceptron import MultiLayerPerceptron
import matplotlib.pyplot as plt

mlp = MultiLayerPerceptron(layers=[2,2,1])
print("Training Neural Network as an XOR Gate")

mse_values = []
for i in range(3000):
    mse = 0.0
    mse += mlp.back_propagation([0,0], [0])
    mse += mlp.back_propagation([0,1], [1])
    mse += mlp.back_propagation([1,0], [1])
    mse += mlp.back_propagation([1,1], [0])
    mse = mse / 4

    mse_values.append(mse)
    
    if (i % 100 == 0):
        print(mse)

    
# print generated weights
mlp.print_weights()

# test weights
print("0 0 = {0:.10f}".format(mlp.run([0,0]) [0]))
print("0 1 = {0:.10f}".format(mlp.run([0,1]) [0]))
print("1 0 = {0:.10f}".format(mlp.run([1,0]) [0]))
print("1 1 = {0:.10f}".format(mlp.run([1,1]) [0]))

def plot(mse_values, epochs):
    plt.figure(figsize=(10, 10))
    plt.plot(range(epochs), mse_values, 'o')
    plt.xlabel("Epochs")
    plt.ylabel("MSE Values")
    plt.show()


# plot shows subtle improvement in overcoming error
plot(mse_values, 3000)