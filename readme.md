## Network:

### Weights: 

W is an (n+1)x(n+1) matrix, for n=# neurons

W_ij is the weight of the connection from neuron i to neuron j+1. "Neuron 0" here means the input. 

[[graphics/simple_network.png]]
To build your W and T matrices, consider the example in the image above. Read this as "neuron 1 (leftmost array) takes from input, neuron 2 (inner array) takes from neuron 1, output takes from neuron 2".