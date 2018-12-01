## Network:
A Network is a specific topology of connected Neuron objects. Connections between neurons can be weighted, and also can carry a time delay. 

Once initialized, a Network's primary method is to take in a set of input signals and output a set of output signals. 

### Weights: 
W is an (n+1)x(n+1) matrix

W_ij is the weight of the connection from neuron i to neuron j+1. "Neuron 0" here means the input. 

![Alt text](graphics/simple_network.png "Simple network example")

To build your W and T matrices, consider the example in the image above. Read this as "neuron 1 (leftmost array) takes from input, neuron 2 (inner array) takes from neuron 1, output takes from neuron 2".