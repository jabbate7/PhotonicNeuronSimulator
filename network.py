# A network is a specific topology of Neurons
class Network:
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
neurons=[1,2,3]
# Weight matrix
# 1 goes to 2, 2 goes to 3, 3 goes to output
W=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
# Time delay matrix
# No time delays
T=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
dt=1

net = Network()
