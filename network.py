from neuron import Neuron
import numpy as np

#Gerrythoughts
# delays should be in units of time, not in units of dt, since should be able to change dt without changing topography of netowrk itself
#also might use solver with variable timestep for next iteration

#history is a list, should it be a np.array?

#lots of for loops

#network only works with 1 dimensional neurons
#one variable is always output, but the non-identity neurons have dimension>=2
# I dont know whether we need to keep track of these in the history vector, probably not
#possible solution is to always have first element of y be the state variable, and store that in ``history''
# also will assume thats the only thing we ever care about then, and can forget about other variables



# A network is a specific topology of Neurons
class Network:
    """
    This is a Network object
    Initiatlize it as 
    mynetwork = Network(neurons, weights, delays, dt)
    neurons is a list of Neuron objects
    weights is a np.array matrix of weights
    delays is a np.array matrix of time delays, in absolute time
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
            #delays is a list of times, need to convert to units of dt
            # divide by dt, so assumes its an np.array and not just a list
            self.delays=(delays/self.dt).astype(int)
        else:
            self.delays = np.zeros((self.num_neurons,self.num_neurons),dtype=int)
        
        max_times = np.apply_along_axis(lambda x: max(x)+1, 0, self.delays)
        #calculates longest time each neurons output is delayed
#        import pdb; pdb.set_trace()
        # set up each neuron with dt and necessary history
        for i,neuron in enumerate(neurons):
            #each neuron needs to store history=longest its output gets delayed
            neuron.set_history(max_times[i])
            neuron.set_dt(self.dt)
            
    def generate_neuron_inputs(self, external_inputs):
        """
        Builds input array for next network step.
        Each element in the array corresponds to a neuron in the network.
        Calculated by summing weighted and delayed inputs to that neuron
        For each element, first add weighted external inputs
        Then add appropriatly delayed weighted outputs from other neurons.        
        """
        def get_prev_output(row, col):
            if col < self.num_inputs:
                return external_inputs[col]
            else:
                col = col - self.num_inputs
#                import pdb; pdb.set_trace()
                return self.neurons[col].hist[self.delays[row][col]] #assumes 1d hist
        
        # an internal function to get the inputs needed to 
        # update neuron states 
        inputs=np.zeros(self.num_neurons)
        for row in range(self.num_neurons):
            for col in range(self.num_neurons+self.num_inputs):
                if self.weights[row][col] != 0:
                    inputs[row] += self.weights[row][col]*get_prev_output(row,col)
        return inputs
                
    def network_step(self,external_inputs):
        """
        Steps the network forward in time by dt.
        Needs the set of current external inputs
        Each Neuron updates based on its state and its input
        """
        external_inputs = np.atleast_1d(external_inputs)
        msg="Please specify {} inputs in an array".format(self.num_inputs)
        assert (len(external_inputs)==int(self.num_inputs)), msg
        # update the state of each neuron 
        neuron_inputs = self.generate_neuron_inputs(external_inputs)
        #neuron_outputs = np.zeros(self.num_neurons)
        for i,neuron in enumerate(self.neurons):
            neuron.step(neuron_inputs[i]) #State is stored internally in each neuron so dont need to store
        #    neuron_outputs[i]=neuron.step(neuron_inputs[i])
        #return (external_inputs, neuron_outputs)

    def __repr__(self):
        return '{}-input, {}-neuron network'.format(self.num_inputs, self.num_neurons)

    def return_states(self):
        """
         Return the state of the network (by querying each neuron
         Optional flag to return just output state or internal state of neuron as well?
         """
        states = np.zeros(self.num_neurons)
        for i,neuron in enumerate(self.neurons):
            states[i]=neuron.y #will only return something useful for 1d neurons currently
        return states
    
    def network_solve(self,external_inputs):
        """
        Solve the network given an array of time dependent external inputs
        External inputs is a num_timesteps X num_inputs array
        outputs a  num_timesteps X num_neuron array which is the network state at each timestep
        """
        #skeleton version for now
        Len_t=external_inputs.shape[0]#first dimension is time
        Net=np.zeros(Len_t, self.num_neurons)
        Net[0, :]=self.return_states()
        for i in range(Len_t-1):
            self.network_step(external_inputs[i, :].squeeze()) #step network forward
            Net[i, :]=self.return_states()
        return Net



    def visualize(self):
        # visualize the network
        pass
