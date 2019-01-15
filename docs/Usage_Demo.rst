
Usage Demonstration Notebook
============================

In this notebook we demonstrate the usage of our neuron and network
classes, calculating and visualizing the dynamics of a single neuron and
a simple network. Further examples can be seen in ``Neuron_plots.ipynb``
and ``Network_plots.ipynb`` in the project github. The preamble to set
up this notebook is below, note that ``Network`` and ``Neuron`` are
imported.

.. code:: ipython3

    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    from neuron import Neuron
    from network import Network
    from IPython.display import HTML

Usage of Neuron Class
=====================

We first study the dynamics of the "Yamada" Laser neuron model, which
describes a two section laser containing a gain region and a saturable
absorber. Input optical signals stimulate a photodiode, which controls
the current into the gain region. The equations of motion are:

.. raw:: latex

   \begin{equation} \dot{I}=\kappa(1-G-Q)I+\beta \\ \dot{G}=\gamma(A+i_{in}(t)-G-IG) \\ \dot{Q}=\gamma(B-Q-aIQ) \end{equation}

Where :math:`I` is the dimensionless laser intensity, and :math:`G` and
:math:`Q` are the inversions of the gain and saturable absorbing media.
:math:`i_{in}` sufficient to produce :math:`G+Q>1` results in a sharp
laser pulse and the system then refracts. For excitability, we require
:math:`\gamma\ll\kappa`: the field intensity is our fast "state"
variable and the inversions are slow "recovery" variables, like the
membrane potential and ion permeability respectively in biological
neurons.

We begin by choosing our model parameters, contained in the dict
``Y1mpars`` below, and then the parameters for the neuron itself,
``Y1params``. The equations of motion above are contained in
``models.Yamada_1``, so we specify this, as well as ``mpars`` and the
timestep ``dt``. It is important that we initialize our neuron in its
steady state, so we use the neuron member function ``steady_state`` to
compute the steady state for our specific choice of model parameters. To
be successful, this requires an initial guess of the steady state, which
is ``y1_steady_est``. We then update the full set of parameters
``Y1params`` for the Neuron object we will be creating.

.. code:: ipython3

    #Create a basic Yamada Neuron 
    #these are the model parameters
    Y1mpars={"a": 2, "A": 6.5, "B":-6., "gamma1": 1,
             "gamma2": 1, "kappa": 50, "beta": 5e-1 }
    
    #neuron parameters
    Y1params={"model" : "Yamada_1","dt": 1e-2, 'mpar': Y1mpars}
    
    #quick estimate of steady state
    y1_steady_est=[Y1mpars['beta']/Y1mpars['kappa'],
                   Y1mpars['A'],Y1mpars['B'] ]
    
     #compute true steady state
    y1_steady=Neuron(Y1params).steady_state(y1_steady_est)
    #add steady state to model parameters
    Y1params["y0"]=y1_steady 
    #now just use Y1params to initialize neurons

We now proceed to initialize a Neuron with
``neuron_object=Neuron(neuron_parameters)``. We then construct an input
signal ``x1`` which is a series of Gaussian inputs, the first of which
is below threshold and will not cause the neuron to spike, and the last
is very noisy but will. The Neuron dynamics in response to this signal
are then computed with ``output=neuron_object.solve(input)``

.. code:: ipython3

    #initialize neuron
    Y1Neuron=Neuron(Y1params)
    
    #create time signal
    t1_end=9. #final time point
    N1=int(np.ceil(t1_end/Y1Neuron.dt)) #this many points
    time1=np.linspace(0.,(N1-1)*Y1Neuron.dt, num=N1 )
    
    #normalized guassian for constructing input signals
    Gaussian_pulse= lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) 
        / (2 * np.power(sig, 2.)))/(np.sqrt(2*np.pi)*sig)
    #create input signal x1
    x1=np.zeros(N1)
    x1+=0.2*Gaussian_pulse(time1, 0.1, 1.e-2)
    x1+=0.5*Gaussian_pulse(time1, 2., 1.e-2)
    x1+=0.5*Gaussian_pulse(time1, 6.5, 5.e-2)*np.random.normal(1,1,N1)
    
    #solve
    y_out1=Y1Neuron.solve(x1)

These results are visualized with
``figure=neuron_object.visualize_plot(input, output, time, steady_state)``.

The upper axis contains the input current to the neuron, and the lower
is the resultant dynamics. The light intensity is the left axis in blue
and the gain and absorber inversions are in red and green on the right
axis. The steady states are also indicated with dashed lines. Note that
a spike is not seen for the initial Gaussian input pulse, as its area is
below threshold. The second and third pulses have the same area and thus
produce nearly identical responses, even though the later is quite
noisy. The refractory period can also be seen as the large time it takes
for the inversion variables to recover after each spike.

.. code:: ipython3

    fig1=Y1Neuron.visualize_plot(x1, y_out1, time1, y1_steady)
    #can use returned figure object to customize plot, as below
    fig1.set_size_inches(10, 8, forward=True)



.. image:: Usage_Demo_files/Usage_Demo_9_0.png


Usage of Network Class
======================

We next consider an inhibitory network of two neurons, each with their
own input channel. Neuron 2 is inhibitively connected to neuron 1: when
it fires it prevents Neuron 1 from firing. These simple networks often
govern reflex behaviors such as the knee-jerk: When the knee is tapped,
the patellar sensory neuron fires, this inhibits a motor neuron
controlling the flexor hamstring muscle, causing it to relax and
allowing your leg to kick out.

We first construct a list of 2 identical neurons
(``neurons=[Neuron(Y1params), Neuron(Y1params)]``) with the same
parameters as the original neuron studied above. We then define our
weight and delay matrices (``weights=np.array(...)``,
``delays=np.array(...)``), and use these to create a network:
``network=Network(neurons, weights, delays)``. The structure of the
weight and delay matrices are discussed further in the "Defining Network
Connections" section of this documentation.

.. code:: ipython3

    # Inhibitory 2 input 2 neuron network
    #list of 2 neurons
    neurons=[Neuron(Y1params), Neuron(Y1params)]
    #neuron 1 receieves input,feeds to neuron 2
    weights=np.array([[1.,0.,0., -0.2],[0.,1.,0., 0.]])
    #Delay on signal from neuron 1 to neuron 2
    delays=np.array([[0., 0.5], [0., 0.]])
    
    #create network
    network2=Network(neurons, weights, delays, dt=0.001)

Since our network accepts two inputs, our input signal is now a 2-D
numpy array, with each column corresponding to a different input
channel. For a given set of input signals, the network dynamics are
calculated with ``output=network.network_solve(input)``. The Network
class also has a member function which computes the total time-dependent
input (sum of internal and external) to each neuron, to better
understand and visualize the network dynamics, this is done via
``total_input=network.network_inputs(output, input)``. Note that the
external inputs are the second argument.

.. code:: ipython3

    
    t2_end=29.
    N2=int(np.ceil(t2_end/network2.dt)) #this many points
    time2=np.linspace(0.,(N2-1)*network2.dt, num=N2 )
    
    in2=np.zeros([N2, 2])
    #scale with gamma1 so drive in units of A
    #drive neuron 1 continuously just above threshold
    in2[:, 0]+=(0.3)*np.heaviside(time2, 0.5)
    #Drive neuron 2 for a short period then turn off
    in2[:, 1]+=(0.6)*np.heaviside(time2-8., 0.5)
    in2[:, 1]+=(-0.6)*np.heaviside(time2-19., 0.5)
    
    #solve network
    output2=network2.network_solve(in2)
    #compute inputs
    input2=network2.network_inputs(output2, in2)

The resultant dynamics are plotted below
via:\ ``figure=network_object.visualize_plot(input, output, time)``.

The upper axes contains the total (weighted, delayed, and summed) input
to each neuron as a function of time, and the lower axes the state of
each neuron (dimensionless laser intensity). Note that once neuron 2
starts firing, neuron 1 stops because neuron 2 inputs a large negative
spike to neuron 1.

.. code:: ipython3

    #use visualize_plot quickly plot of the network dynamics
    fig2=network2.visualize_plot(input2, output2, time2)
    fig2.set_size_inches(10, 8, forward=True)



.. image:: Usage_Demo_files/Usage_Demo_16_0.png


Below is a visualization of the same dynamics as an animated graph,
generated using the member function ``visualize animation``. To see the
resultant animation, We need to call
``HTML(animation.to_`.to_html5_video())`` where ``HTML`` was imported
from ``IPython.display``

Each neuron is depicted as a node of the network which brightens when it
fires. The connectivity between network elements and their relative
strengths are also indicated.

.. code:: ipython3

    %%capture 
    an2 = network2.visualize_animation(inputs=in2, outputs=output2);
    #create animation
    #capture is to supress output,
    #remove to generate a static image of the network

.. code:: ipython3

    #view animation
    HTML(an2.to_html5_video()) 
    #note that this HTML call can be time-consuming




.. raw:: html

    <video width="432" height="432" controls autoplay loop>
      <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAACl/W1kYXQAAAKvBgX//6vcRem9
    5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTUyIHIyODU0IGU5YTU5MDMgLSBILjI2NC9NUEVHLTQg
    QVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDE3IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcv
    eDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9
    MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVm
    PTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6
    b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MTIgbG9v
    a2FoZWFkX3RocmVhZHM9MiBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxh
    Y2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHly
    YW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3
    ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTI1IHNjZW5lY3V0PTQwIGludHJhX3JlZnJl
    c2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAg
    cXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAIlWWI
    hAAn//71sXwKasnzigzoMi7hlyTJrrYi4m0AwAAAAwAFUEq0Xzg3/fjOAADqgAP0gIZ2NBVLTADd
    bJhz+Zjgp7dAWowmOaeJwckVzPDxZqpy9pkAGsX6Y6aq633xSQLzSlJ5BSDtMT4KZJ97RHFNhV7j
    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
    ZHRzAAAAHGVsc3QAAAAAAAAAAQAAJxAAAAEAAAEAAAAAL9xtZGlhAAAAIG1kaGQAAAAAAAAAAAAA
    AAAAADIAAAH0AFXEAAAAAAAtaGRscgAAAAAAAAAAdmlkZQAAAAAAAAAAAAAAAFZpZGVvSGFuZGxl
    cgAAAC+HbWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAAJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAA
    DHVybCAAAAABAAAvR3N0YmwAAACzc3RzZAAAAAAAAAABAAAAo2F2YzEAAAAAAAAAAQAAAAAAAAAA
    AAAAAAAAAAABsAGwAEgAAABIAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAY//8AAAAxYXZjQwFkAB//4QAYZ2QAH6zZQbDehAAAAwAEAAADAyA8YMZYAQAGaOvjyyLAAAAA
    HHV1aWRraEDyXyRPxbo5pRvPAyPzAAAAAAAAABhzdHRzAAAAAAAAAAEAAAPoAAAAgAAAACBzdHNz
    AAAAAAAAAAQAAAABAAAA+wAAAfUAAALvAAAecGN0dHMAAAAAAAADzAAAAAEAAAEAAAAAAQAAAoAA
    AAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAA
    AAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAA
    AQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAAB
    AAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEA
    AAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAA
    AoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAA
    gAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAA
    AAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAIAAAAAAgAAAIAA
    AAACAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAgAAAQAAAAABAAACgAAA
    AAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAA
    AQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAAB
    AAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEA
    AAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAA
    AQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAAC
    gAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACA
    AAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAA
    AAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAA
    AAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAA
    AQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAGA
    AAAAAQAAAIAAAAABAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAwAAAQAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAAB
    AAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAA
    AAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEA
    AAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAA
    AAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAA
    AAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAA
    AQAAAIAAAAABAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAAB
    AAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAIAAAAAAgAAAIAAAAACAAAB
    AAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAgAAAQAAAAABAAACgAAAAAEAAAEA
    AAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAA
    AAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAA
    AAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAA
    AQAAAIAAAAABAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAgAAAQAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAYAAAAAB
    AAAAgAAAAAEAAAEAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAADAAABAAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAA
    AAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEA
    AAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAA
    AAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAA
    AAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAA
    AQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAAB
    AAAAAAAAAAEAAACAAAAAAQAAAgAAAAACAAAAgAAAAAEAAAEAAAAAAQAAAoAAAAABAAABAAAAAAEA
    AAAAAAAAAQAAAIAAAAACAAABAAAAAAEAAAGAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAAB
    AAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAMA
    AAEAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAACAAABAAAAAAEAAAKAAAAAAQAA
    AQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAAC
    gAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACA
    AAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAA
    AAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAA
    AAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAA
    AQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAgAAAAACAAAAgAAAAAIAAAEAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAACAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAAB
    AAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAgAAAAACAAAAgAAAAAEAAAEAAAAAAQAA
    AoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAACAAABAAAAAAEAAAGAAAAAAQAAAIAAAAABAAAC
    gAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACA
    AAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAA
    AAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAA
    AAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAA
    AQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAAB
    AAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAACAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAgAAAQAAAAABAAAC
    gAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACA
    AAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAA
    AAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAA
    AAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAA
    AQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAAB
    AAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEA
    AAGAAAAAAQAAAIAAAAACAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAgAA
    AQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAA
    AAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEA
    AAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAA
    AAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAA
    AAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAA
    AQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAAB
    AAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEA
    AAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAA
    AoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAA
    gAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAA
    AAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAA
    AAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAA
    AAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAA
    AQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAAB
    AAAAgAAAAAMAAAEAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAACAAABAAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAA
    AAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEA
    AAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAYAA
    AAABAAAAgAAAABxzdHNjAAAAAAAAAAEAAAABAAAD6AAAAAEAAA+0c3RzegAAAAAAAAAAAAAD6AAA
    C0wAAAC3AAAAIQAAAB8AAAAZAAAAIQAAABYAAAAVAAAAEwAAABkAAAARAAAAEAAAABAAAAAZAAAA
    EQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQ
    AAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEA
    AAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAA
    ADcAAAATAAAAFQAAAE0AAAB8AAAAyAAAAMUAAABGAAAAXAAAAFEAAABhAAAAVgAAABoAAAATAAAA
    EAAAAC0AAAAUAAAAEAAAABIAAAAkAAAAFQAAABMAAAASAAAAIgAAABEAAAAQAAAAEAAAABkAAAAR
    AAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAA
    AAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAA
    ABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAA
    GQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQ
    AAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkA
    AAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAADwAAAAUAAAATwAA
    ATIAAABmAAAAPAAAAJQAAACbAAAAdAAAAGkAAABCAAAAGQAAABIAAAASAAAAKwAAABUAAAAUAAAA
    EAAAACIAAAARAAAAEAAAABAAAAAkAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAAR
    AAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAA
    AAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAA
    ABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAA
    GQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAApLAAAAKgAAAB0AAAAaAAAAGQAAACsAAAAc
    AAAAFgAAABUAAAAlAAAAFgAAABYAAAATAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAA
    AAAZAAAAEQAAABAAAAAQAAABGwAAABkAAAAZAAAAQAAAAFoAAAECAAAAxAAAACAAAABJAAAAUwAA
    AGIAAABaAAAAHQAAABcAAAASAAAAHwAAABUAAAAUAAAAEgAAABkAAAARAAAAEAAAABAAAAAZAAAA
    EQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAADEAAAARAAAAEAAAABEAAABMAAABOQAAAGUAAAA6
    AAAArgAAAK4AAABuAAAAjgAAAB4AAAATAAAAFwAAAB0AAAAYAAAAFgAAABMAAAAZAAAAFQAAABMA
    AAATAAAAGQAAABUAAAATAAAAEwAAABkAAAAVAAAAEwAAABMAAAAZAAAAFQAAABMAAAATAAAAGQAA
    ABUAAAATAAAAEwAAABkAAAAVAAAAEwAAABMAAAAZAAAAFQAAABMAAAATAAAAGQAAABUAAAATAAAA
    EwAAABkAAAAVAAAAEwAAABMAAAAZAAAAFQAAABMAAAATAAAAGQAAABUAAAATAAAAEwAAABkAAAAV
    AAAAEwAAABMAAAA6AAAAEwAAAFQAAAEuAAAAUwAAACwAAAB2AAAAmgAAAHkAAABeAAAAXgAAAB0A
    AAAZAAAAEQAAACUAAAAVAAAAEwAAABEAAAAZAAAAEwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAA
    ABkAAAATAAAAEQAAABEAAAAZAAAAEwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAAABkAAAATAAAA
    EQAAABEAAAAZAAAAEwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAAABkAAAATAAAAEQAAABEAAAAZ
    AAAAEwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAAADYAAAATAAAAEQAAAF0AAAEEAAAAhgAAADUA
    AAA+AAAAaQAAAGAAAACBAAAAGAAAACEAAAAUAAAAEgAAABEAAAAZAAAAEQAAABAAAAAQAAAAGQAA
    ABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAA
    EAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAAR
    AAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAA
    AAAgAAAJ+AAAAGkAAADCAAAA4gAAADEAAAAtAAAATQAAAGYAAAB4AAAAGQAAABYAAAAQAAAAJwAA
    ABMAAAAQAAAAEAAAACoAAAAUAAAAEAAAABAAAAApAAAAFAAAABEAAAAQAAAAKgAAABQAAAARAAAA
    EAAAACkAAAAUAAAAEQAAABAAAAAqAAAAFAAAABEAAAAQAAAAKQAAABQAAAARAAAAEAAAACoAAAAU
    AAAAEQAAABAAAAAxAAAAFAAAABEAAAARAAAAJgAAABQAAAARAAAAEQAAACcAAAAUAAAAEQAAABEA
    AAAlAAAAFAAAABEAAAARAAAAQgAAABMAAAARAAAAUQAAAJcAAABSAAAA2wAAACoAAAAtAAAAcwAA
    AHsAAABiAAAAHgAAABYAAAAWAAAALwAAABcAAAAVAAAAFQAAACsAAAAXAAAAFQAAABUAAAAmAAAA
    FwAAABQAAAAUAAAAKgAAABUAAAAUAAAAFAAAACYAAAAVAAAAFAAAABQAAAAqAAAAFQAAABQAAAAU
    AAAAJgAAABUAAAAUAAAAFAAAACoAAAAVAAAAFAAAABQAAAAmAAAAFQAAABQAAAAUAAAAJwAAABUA
    AAAUAAAAFAAAACYAAAAVAAAAFAAAABQAAAAlAAAAFQAAABQAAAAUAAAATgAAABUAAAAWAAAAZAAA
    ARkAAAB3AAAALgAAAEcAAAB2AAAAbgAAAIMAAAAZAAAAOQAAABgAAAAVAAAAEQAAAC8AAAATAAAA
    EQAAABEAAAAoAAAAEwAAABEAAAAQAAAALAAAABMAAAASAAAAEgAAACgAAAATAAAAEgAAABIAAADh
    AAAAGwAAABkAAAASAAAAKQAAABMAAAASAAAAEgAAACkAAAATAAAAEgAAABIAAAAoAAAAEwAAABIA
    AAASAAAAKQAAABMAAAASAAAAEgAAACgAAAATAAAAEgAAABIAAAApAAAAEwAAABIAAAASAAAAKAAA
    ABMAAAASAAAAEgAAACkAAAATAAAAEgAAABIAAAAoAAAAEwAAABIAAAASAAAAKQAAABMAAAASAAAA
    EgAAACgAAAATAAAAEgAAABIAAAApAAAAEwAAABIAAAASAAAAKAAAABMAAAASAAAAEgAAACkAAAAT
    AAAAEgAAABIAAAAoAAAAEwAAABIAAAASAAAAKQAAABMAAAASAAAAEgAAACgAAAATAAAAEgAAABIA
    AAApAAAAEwAAABIAAAASAAAAKAAAABMAAAASAAAAEgAAACkAAAATAAAAEgAAABIAAAA0AAAAEwAA
    ABIAAAASAAAATQAAAF8AAAEvAAAAwQAAADgAAAAzAAAAdwAACjgAAABmAAAAJAAAABoAAAAbAAAA
    MwAAAB4AAAAdAAAAFQAAACkAAAAWAAAAEwAAABMAAAAhAAAAFgAAABMAAAATAAAAHAAAABUAAAAT
    AAAAEQAAABkAAAATAAAAEQAAABEAAAAZAAAAEwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAAABkA
    AAATAAAAEQAAABEAAAAZAAAAEwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAAABkAAAATAAAAEQAA
    ABEAAAAZAAAAEwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAAABkAAAATAAAAEQAAABEAAAAZAAAA
    EwAAABEAAAARAAAAGQAAABMAAAARAAAAEQAAABkAAAATAAAAEQAAABEAAAAZAAAAEwAAABEAAAAR
    AAAAGQAAABMAAAARAAAAEQAAABkAAAATAAAAEQAAABEAAAAZAAAAEwAAABEAAAARAAAAGQAAABMA
    AAARAAAAEQAAABkAAAATAAAAEQAAABEAAAAZAAAAEwAAABEAAAARAAAAPgAAABQAAABJAAAAcwAA
    AHoAAAC/AAAAVgAAAFgAAABcAAAAbAAAAEsAAAAdAAAAFwAAABYAAAAtAAAAFwAAABUAAAATAAAA
    JQAAABcAAAASAAAAEwAAABwAAAAVAAAAEgAAABAAAAAZAAAAFAAAABAAAAAQAAAAGQAAABQAAAAQ
    AAAAEAAAABkAAAAUAAAAEAAAABAAAAAZAAAAFAAAABAAAAAQAAAAGQAAABQAAAAQAAAAEAAAABkA
    AAAUAAAAEAAAABAAAAAZAAAAFAAAABAAAAAQAAAAGQAAABQAAAAQAAAAEAAAABkAAAAUAAAAEAAA
    ABAAAAAZAAAAFAAAABAAAAAQAAAAGQAAABQAAAAQAAAAEAAAABkAAAAUAAAAEAAAABAAAAAZAAAA
    FAAAABAAAAAQAAAAGQAAABQAAAAQAAAAEAAAABkAAAAUAAAAEAAAABAAAAAZAAAAFAAAABAAAAAQ
    AAAAGQAAABQAAAAQAAAAEAAAABkAAAAUAAAAEAAAABAAAAAZAAAAFAAAABAAAAAQAAAAGQAAABQA
    AAAQAAAAEAAAACUAAAAUAAAAEAAAABAAAAA5AAAATAAAAIgAAAA1AAAAygAAAEwAAABJAAAAbwAA
    AGYAAABUAAAAGgAAABMAAAAQAAAAJAAAABMAAAATAAAAEgAAABkAAAARAAAAEAAAABAAAAAZAAAA
    EQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAYAAAAEQAAABAAAAAQ
    AAAAGAAAABAAAAAUc3RjbwAAAAAAAAABAAAALAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIA
    AAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABM
    YXZmNTcuODMuMTAw
    ">
      Your browser does not support the video tag.
    </video>


