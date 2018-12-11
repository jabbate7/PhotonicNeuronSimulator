from neuron import Neuron
import numpy as np

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
        
    def __init__(self, neurons, weights, delays=None, dt=None):
        self.neurons=neurons
        self.weights=weights
        if (dt is not None):
            self.dt=dt
        else:
            self.dt=neurons[0].dt
        
        self.num_neurons = np.shape(weights)[0]
        self.num_inputs = np.shape(weights)[1]-self.num_neurons

        if (delays is not None):
            self.delays=delays
        else:
            self.delays = np.zeros((self.num_neurons,self.num_neurons),dtype=int)
        
        max_times = np.apply_along_axis(lambda x: max(x)+1, 0, self.delays)
#        import pdb; pdb.set_trace()
        # set up each neuron with dt and necessary history
        for i,neuron in enumerate(neurons):
            neuron.set_history(max_times[i])
            neuron.set_dt(self.dt)
            
    def generate_neuron_inputs(self, external_inputs):
        def get_prev_output(row, col):
            if col < self.num_inputs:
                return external_inputs[col]
            else:
                col = col - self.num_inputs
#                import pdb; pdb.set_trace()
                return self.neurons[col].hist[self.delays[row][col]]
        
        # an internal function to get the inputs needed to 
        # update neuron states 
        inputs=np.zeros(self.num_neurons)
        for row in range(self.num_neurons):
            for col in range(self.num_neurons+self.num_inputs):
                if self.weights[row][col] != 0:
                    inputs[row] += self.weights[row][col]*get_prev_output(row,col)
        return inputs
                
    def network_step(self,external_inputs):
        external_inputs = np.atleast_1d(external_inputs)
        msg="Please specify {} inputs in an array".format(self.num_inputs)
        assert (len(external_inputs)==int(self.num_inputs)), msg
        # update the state of each neuron 
        neuron_inputs = self.generate_neuron_inputs(external_inputs)
        neuron_outputs = np.zeros(self.num_neurons)
        for i,neuron in enumerate(self.neurons):
            neuron_outputs[i]=neuron.step(neuron_inputs[i])
        return (external_inputs, neuron_outputs)

    def __repr__(self):
        return '{}-input, {}-neuron network'.format(self.num_inputs, self.num_neurons)

    def return_states(self):
        # return the state of the network (by querying each neuron)
        pass
    
    def visualize(self):
        # visualize the network
        pass
