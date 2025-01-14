from perceptron import MultiLayerPerceptron

mlp = MultiLayerPerceptron(layers=[2, 2, 1])
mlp.set_weights([[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]])

mlp.printWeights()

print("MLP:")
print("0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
print("0 1 = {0:.10f}".format(mlp.run([0, 1])[0]))
print("1 0 = {0:.10f}".format(mlp.run([1, 0])[0]))
print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))


'''
Layer 2 Neuron 0 [-10 -10  15]
Layer 2 Neuron 1 [ 15  15 -10]
Layer 3 Neuron 0 [ 10  10 -15]

MLP:
0 0 = 0.0066958493
0 1 = 0.9923558642
1 0 = 0.9923558642
1 1 = 0.0071528098
'''