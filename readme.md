Network:

Weights: 

W is an (n+1)x(n+1) matrix, for n=# neurons

W_ij is the weight of the connection from neuron i to neuron j+1. "Neuron 0" here means the input. 

<Input picture of matrix with "0, 1, 2" going down left, "1, 2, 3" on right.>
Consider the example: [[1,0,0],[0,1,0],[0,0,1]]
Read this as "neuron 1 (leftmost array) takes from input, neuron 2 (inner array) takes from neuron 1, output takes from neuron 2".