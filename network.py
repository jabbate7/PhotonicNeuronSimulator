from neuron import Neuron
from scipy.sparse import csr_matrix

# A network is a specific topology of Neurons
class Network:
    """
    This is a Network object
    Initiatlize it as 
    mynetwork = Network(neurons, weights, delays, dt)
    neurons is an array of Neuron objects
    weights is a matrix of weights
    delays is a matrix of time delays
    dt is the time step for propagating the network
    see the readme for structure of "weights" and "delays"
    """
        
    def __init__(self, neurons, weights, delays, dt):
        self.neurons=neurons
        self.weights=weights
        self.delays=delays
        self.dt=dt
        
        self.num_neurons = np.shape(weights)[0]
        self.num_inputs = np.shape(weights)[1]-self.num_neurons
        max_times = np.apply_along_axis(lambda x: max(x), 0, delays)
        
        # set up each neuron with dt and necessary history
        for i,neuron in enumerate(neurons):
            neuron.set_history(max_times[i])
            neuron.set_dt(self.dt)
            
    def generate_neuron_inputs(external_inputs, self):
        def get_prev_output(row, col):
            if col < self.num_inputs:
                return external_inputs[col]
            else:
                return self.neurons[row].hist(self.delays[row][col+self.num_inputs])
        
        # an internal function to get the inputs needed to 
        # update neuron states 
        inputs=np.zeros(num_neurons)
        for row in self.weights:
            for col in self.weights:
                if self.weights[row][col] != 0:
                    inputs[row] += self.weights[row][col]*get_prev_output(row,col)
        return inputs
                
    def propagate(external_inputs, self):
        # update the state of each neuron 
        neuron_inputs = generate_neuron_inputs(external_inputs)
        for i,neuron in enumerate(self.neurons):
            neuron_outputs[i]=neuron.step(neuron_inputs[i])
        return (external_inputs, neuron_outputs)

    def return_states(self):
        # return the state of the network (by querying each neuron)
        pass
    
    def visualize(self):
        # visualize the network
        pass
        
# List of Neuron objects
neur_1=Neuron({'x0': 0, })
neurons=[neur_1,neur_2,neur_3]
# Weight matrix
# Input goes 1, 1 goes to 2, 2 goes to 3, 3 goes to output
W=[[1,0,0],[0,1,0]]
# Time delay matrix
# No time delays
T=[[1,0,0],[0,10,0]]
dt=1

net = Network()
