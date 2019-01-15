Profiling
-----------

We used the ``cProfile`` utility to profile our neural network solver. Our benchmarking was done for a 4-input 16-neuron network with random weights and delays. Our input was a time series of length 20000.


.. figure:: graphics/profiling.png
   :align: center

   Profiling results

The entire simulation takes about 14 seconds to run. Each of the ``network_step()`` and ``generate_neuron_inputs()`` functions is called 19999 times (the number of time steps necesary to move through the time series). The most time consuming step was in the ``generate_neuron_inputs()`` function. 

This function calculates the sum of all weighted and delayed inputs to each neuron in the network. By necessity, it must loop over all neurons on the network, annd iterate over all entries in the adjacency matrix of connections. Further algorithmic optimization is not evident.