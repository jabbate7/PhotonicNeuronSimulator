Key decisions
-------------

In this section, we describe three instances where we had to make careful design decisions in the implementation of the project. In our excitement to see the physics, it was often tempting to take the easy route and write hacky and poorly-thought out code. The lessons we've learned in APC524 of keeping the code modular, test-driven and object-oriented were invaluable in helping us avoid some pitfalls.

#. **Should** ``neuron`` **should be its own class at all?** The neuron is fundamentally just an ODE solver for a system of equations :math:`y(t) = f(\dot{y}(t), x(t))`. Given a model :math:`f`, and input :math:`x(t)`, it should return the output :math:`y(t)`. This can be implemented as a function, without any need for defining an object. However, since conceptually the neuron is a building block for the entire system, we felt that it needed to have a respectable existence as an object of its own accord. This paid off later, when we wanted to add more bells and whistles, such as the ability for each neuron to store its own history, calculate its own steady state, and so on, all of which would have been impossible had the ``neuron`` been implemented just as a function.

#. **Should the visualizers be their own class?** While the solving of the equations is an integral part of our project, being able to visualize the results is equally important. There are potentially many ways of visualizing a neural network in action, such as input and output traces over time, phase space plots, graphical representations of the nodes and so on. One possibility was to have a visualizer class, with dedicated functions to provide the desired kind of visualization on the desired part of the network. We instead decided to include functions such as ``visualize_plot()`` in the neuron and network objects instead. Ths gave the neuron and network objects the basic ability to visualize themselves. To accomodate more advanced visualizations in the future, it would be prudent to create a specialized ``visualizer`` class.
    
#. **Who should keep track of history?** An isolated neuron's dynamics only depend on the present values of its inputs, and not on its past inputs or states. The complexity of excitable neural networks arises from the time delays in propagation of signals from one node to another. This turns the network into a set of coupled delayed-diffferential equations. A neuron embedded in a network affects future values of other neurons in the network, as well as itself. This led to an important question. Should the network object keep track of the histories of all its constituent neurons? Or should each neuron keep track of its own history?

    We ultimately decided to have each neuron keep track of its own history via the ``hist`` attribute, even though an isolated neuron object itself has no need to track its own history. When a network object is initialized, it analyzes the graph and calculates how much history each of its constituent neurons needs to store. Each neuron's history is then set using ``Neuron.set_history()``.





