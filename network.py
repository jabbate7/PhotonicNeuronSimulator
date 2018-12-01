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
        # loop over neurons and set dt / feed max time delay
    
    def generate_inputs(self):
        # for each neuron, 
        pass
        
# List of Neuron objects
neur_1=Neuron({'x0': 0, })
neurons=[neur_1,neur_2,neur_3]
# Weight matrix
# Input goes 1, 1 goes to 2, 2 goes to 3, 3 goes to output
W=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
# Time delay matrix
# No time delays
T=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
dt=1

net = Network()
