
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
    Y1mpars={"a": 2, "A": 6.5, "B":-6., "gamma1": 1,
             "gamma2": 1, "kappa": 50, "beta": 5e-1 }#these are the model parameters
    
    Y1params={"model" : "Yamada_1","dt": 1e-2, 'mpar': Y1mpars} #neuron parameters
    
    y1_steady_est=[Y1mpars['beta']/Y1mpars['kappa'],
                   Y1mpars['A'],Y1mpars['B'] ] #quick estimate of steady state
    
    y1_steady=Neuron(Y1params).steady_state(y1_steady_est) #compute true steady state
    
    Y1params["y0"]=y1_steady #change model parameters so that starts w this ss
    
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
    x1+=0.5*Gaussian_pulse(time1, 6.5, 5.e-2)*np.random.normal(1, 1,N1)
    
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
    #neuron 1 is regularly firing, neuron 2 stops neuron 1 from firing 
    
    neurons=[Neuron(Y1params), Neuron(Y1params)] #list of 2 neurons
    weights=np.array([[1.,0.,0., -0.2],[0.,1.,0., 0.]])#neuron 1 receieves input,feeds to neuron 2
    delays=np.array([[0., 0.5], [0., 0.]])#Delay on signal from neuron 1 to neuron 2
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

    #use visualize_plot to generate a quick plot of the network dynamics
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
    an2 = network2.visualize_animation(inputs=in2, outputs=output2);#create animation
    #capture is to supress output, remove to generate a static image of the network


.. code:: ipython3

    #view animation
    HTML(an2.to_html5_video()) #note that this HTML call can be time-consuming




.. raw:: html

    <video width="432" height="432" controls autoplay loop>
      <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAACnxm1kYXQAAAKvBgX//6vcRem9
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
    cXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAJZGWI
    hAAn//71sXwKasnzigzoMi7hlyTJrrYi4m0AwAAAAwAFUEq0Xzg3/fjOAADqgAP0gIZ2NBVLTADd
    bJhz+Zjgp7dAWowmOaeJwckVzPDxZqpy9pkAGsX6Y6aq633xSQLzSlJ5BSDtMT4KZJ97RHFNhV7j
    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
    cwAAAAAAAAAEAAAAAQAAAPsAAAH1AAAC7wAAHnBjdHRzAAAAAAAAA8wAAAABAAABAAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAAB
    AAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAA
    AAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACAAAAAAIAAACA
    AAAAAgAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAIAAAEAAAAAAQAAAoAA
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
    AAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEA
    AAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAA
    AQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAAB
    gAAAAAEAAACAAAAAAQAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAMAAAEA
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
    AAEAAACAAAAAAQAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAA
    AQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACAAAAAAIAAACAAAAAAgAA
    AQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAIAAAEAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAIAAAEAAAAA
    AQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAAB
    AAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEA
    AAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAA
    AQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAAC
    gAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACA
    AAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAA
    AAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAA
    AAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAGAAAAA
    AQAAAIAAAAABAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAwAAAQAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
    AAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAA
    AAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAA
    AQAAAAAAAAABAAAAgAAAAAEAAAIAAAAAAgAAAIAAAAABAAABAAAAAAEAAAKAAAAAAQAAAQAAAAAB
    AAAAAAAAAAEAAACAAAAAAwAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAA
    AIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAA
    AAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEA
    AAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAA
    AAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAA
    AAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAA
    AQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAGAAAAAAQAAAIAAAAAC
    AAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAgAAAQAAAAABAAACgAAAAAEA
    AAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAA
    AoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAA
    gAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAA
    AAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAA
    AAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAA
    AAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAA
    AQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAAB
    AAAAgAAAAAEAAAIAAAAAAgAAAIAAAAABAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAwAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAA
    AQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAAC
    gAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACA
    AAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAA
    AAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAA
    AAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAA
    AQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAIAAAAAAgAAAIAAAAABAAABAAAAAAEA
    AAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAwAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
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
    AQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAAB
    AAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAABgAAAAAEA
    AACAAAAAAQAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAIAAAEAAAAAAQAA
    AoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAA
    gAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAA
    AAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAA
    AAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAA
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
    AAABgAAAAAEAAACAAAAAAgAAAQAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAIA
    AAEAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKA
    AAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAA
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
    AQAAAIAAAAADAAABAAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAgAAAQAAAAAB
    AAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEA
    AACAAAAAAQAAAoAAAAABAAABAAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAA
    AAAAAAABAAAAgAAAAAEAAAKAAAAAAQAAAQAAAAABAAAAAAAAAAEAAACAAAAAAQAAAoAAAAABAAAB
    AAAAAAEAAAAAAAAAAQAAAIAAAAABAAACgAAAAAEAAAEAAAAAAQAAAAAAAAABAAAAgAAAAAEAAAGA
    AAAAAQAAAIAAAAAcc3RzYwAAAAAAAAABAAAAAQAAA+gAAAABAAAPtHN0c3oAAAAAAAAAAAAAA+gA
    AAwbAAAAvgAAAB4AAAAcAAAAFQAAACQAAAAXAAAAFgAAABYAAAAZAAAAEQAAABAAAAAQAAAAGQAA
    ABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAA
    EAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAAR
    AAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABwAAAARAAAAEAAAABAA
    AAA7AAAAEQAAABIAAABQAAAAiAAAAM8AAADNAAAASAAAAGEAAABPAAAAagAAAGAAAAAbAAAAFQAA
    ABMAAAAfAAAAEwAAABMAAAASAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAA
    EQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQ
    AAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEA
    AAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAA
    ABkAAAARAAAAEAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAA
    EAAAABAAAAAZAAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABkAAAARAAAAEAAAABAAAAAZ
    AAAAEQAAABAAAAAQAAAAGQAAABEAAAAQAAAAEAAAABwAAAARAAAAEAAAABAAAAA+AAAAEAAAAEsA
    AAEuAAAAZQAAADgAAACeAAAAfQAAAG8AAAB0AAAAQwAAABcAAAATAAAAEgAAADAAAAAWAAAAEwAA
    ABAAAAAcAAAAFAAAABIAAAAQAAAAGQAAABQAAAASAAAAEAAAABkAAAAUAAAAEgAAABAAAAAZAAAA
    FAAAABIAAAAQAAAAGQAAABQAAAASAAAAEAAAABkAAAAUAAAAEgAAABAAAAAZAAAAFAAAABIAAAAQ
    AAAAGQAAABQAAAASAAAAEAAAABkAAAAUAAAAEgAAABAAAAAZAAAAFAAAABIAAAAQAAAAGQAAABQA
    AAASAAAAEAAAABkAAAAUAAAAEgAAABAAAAAZAAAAFAAAABIAAAAQAAAAGQAAABQAAAASAAAAEAAA
    ABkAAAAUAAAAEgAAABAAAAAZAAAAFAAAABIAAAAQAAALGwAAADYAAAAcAAAAGwAAABkAAAAnAAAA
    GwAAABUAAAATAAAAIQAAABoAAAAVAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFgAAABEAAAAQ
    AAAAGQAAABUAAAAQAAAAEAAAARsAAAAdAAAAHgAAAEUAAABvAAABCQAAALwAAAAgAAAARgAAAEoA
    AAB1AAAAXwAAACEAAAAVAAAAFwAAACsAAAAdAAAAFgAAABUAAAAcAAAAGwAAABYAAAAVAAAAGQAA
    ABoAAAAUAAAAFAAAABkAAAAaAAAAFQAAABQAAAA0AAAAGgAAABQAAAAUAAAAUAAAAU4AAABzAAAA
    MwAAALQAAACvAAAAcAAAAHcAAAAhAAAAEwAAABIAAAAZAAAAGAAAABMAAAASAAAAGQAAABgAAAAS
    AAAAEgAAABkAAAAYAAAAEwAAABIAAAAZAAAAGAAAABIAAAASAAAAGQAAABgAAAATAAAAEgAAABkA
    AAAYAAAAEgAAABIAAAAZAAAAGAAAABMAAAASAAAAGQAAABgAAAASAAAAEgAAABkAAAAYAAAAEwAA
    ABIAAAAZAAAAGAAAABIAAAASAAAAGQAAABgAAAATAAAAEgAAABkAAAAYAAAAEgAAABIAAAAZAAAA
    GAAAABMAAAASAAAAOQAAABYAAABRAAABKQAAAFkAAAAwAAAAcQAAAIoAAABtAAAAcgAAAEYAAAAa
    AAAAEQAAABAAAAAcAAAAFwAAABEAAAARAAAAGQAAABYAAAARAAAAEAAAABkAAAAVAAAAEAAAABAA
    AAAZAAAAFgAAABEAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAWAAAAEQAAABAAAAAZAAAAFQAA
    ABAAAAAQAAAAGQAAABYAAAARAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFgAAABEAAAAQAAAA
    GQAAABUAAAAQAAAAEAAAABkAAAAWAAAAEQAAABAAAABAAAAAFQAAABQAAABSAAABCgAAAHEAAAAu
    AAAAPAAAAGYAAABbAAAAXgAAADkAAAAbAAAAEwAAABAAAAAZAAAAGAAAABEAAAARAAAAGQAAABgA
    AAATAAAAEQAAABkAAAAYAAAAEQAAABEAAAAZAAAAGAAAABMAAAARAAAAGQAAABgAAAARAAAAEQAA
    ABkAAAAYAAAAEwAAABEAAAAZAAAAGAAAABEAAAARAAAAGQAAABgAAAATAAAAEQAAABkAAAAYAAAA
    EQAAABEAAAAZAAAAGAAAABMAAAARAAAAGQAAABgAAAARAAAAEQAAABkAAAAYAAAAEwAAABEAAAAy
    AAAAFgAACvkAAABpAAAAyAAAANUAAAA9AAAAKwAAAEMAAABjAAAAZAAAAB0AAAARAAAAEAAAAC4A
    AAAVAAAAEAAAABAAAAAnAAAAGAAAABMAAAAQAAAAHwAAABcAAAATAAAAEgAAABkAAAAVAAAAEAAA
    ABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAA
    FQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQ
    AAAAGQAAABUAAAAQAAAAEAAAADUAAAAVAAAAEgAAAEIAAAFIAAAAHwAAACYAAAByAAAAtgAAAHUA
    AABvAAAAXAAAABsAAAAVAAAAEQAAAB4AAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAA
    ABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAA
    EAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAV
    AAAAEAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAAEIAAAAXAAAAFAAAAFIA
    AAD6AAAAewAAADAAAABJAAAAcwAAAGIAAABpAAAAMgAAABsAAAAQAAAAEAAAACQAAAAZAAAAEQAA
    ABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAADRAAAA
    HQAAABAAAAAWAAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQ
    AAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUA
    AAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAA
    ABkAAAAVAAAAEAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAA
    EAAAABAAAAAZAAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAZ
    AAAAFQAAABAAAAAQAAAAGQAAABUAAAAQAAAAEAAAABkAAAAVAAAAEAAAABAAAAAhAAAAFQAAABAA
    AAAQAAAATQAAABUAAABbAAABOgAAAL8AAAA0AAAANAAAAGkAAAsnAAAAZgAAACAAAAAYAAAAFgAA
    ACYAAAAZAAAAFgAAABQAAAAZAAAAFgAAABMAAAATAAAAGQAAABYAAAATAAAAEwAAABkAAAAWAAAA
    EwAAABMAAAAZAAAAFgAAABMAAAATAAAAGQAAABYAAAATAAAAEwAAABkAAAAWAAAAEwAAABMAAAAZ
    AAAAFgAAABMAAAATAAAAGQAAABYAAAATAAAAEwAAABkAAAAWAAAAEwAAABMAAAAZAAAAFgAAABMA
    AAATAAAAGQAAABYAAAATAAAAEwAAABkAAAAWAAAAEwAAABMAAAAZAAAAFgAAABMAAAATAAAAGQAA
    ABYAAAATAAAAEwAAABkAAAAWAAAAEwAAABMAAAAZAAAAFgAAABMAAAATAAAAGQAAABYAAAATAAAA
    EwAAABkAAAAWAAAAEwAAABMAAAAZAAAAFgAAABMAAAATAAAAGQAAABYAAAATAAAAEwAAABkAAAAW
    AAAAEwAAABMAAAAZAAAAFgAAABMAAAATAAAAGQAAABYAAAATAAAAEwAAADwAAAAVAAAARwAAAHYA
    AACJAAAAzwAAAFgAAABkAAAAXQAAAHEAAABLAAAAHAAAABUAAAAQAAAAKAAAABYAAAAQAAAAEAAA
    ACUAAAAWAAAAEwAAABMAAAAZAAAAEwAAABAAAAAQAAAAGQAAABMAAAAQAAAAEAAAABkAAAATAAAA
    EAAAABAAAAAZAAAAEwAAABAAAAAQAAAAGQAAABMAAAAQAAAAEAAAABkAAAATAAAAEAAAABAAAAAZ
    AAAAEwAAABAAAAAQAAAAGQAAABMAAAAQAAAAEAAAABkAAAATAAAAEAAAABAAAAAZAAAAEwAAABAA
    AAAQAAAAGQAAABMAAAAQAAAAEAAAABkAAAATAAAAEAAAABAAAAAZAAAAEwAAABAAAAAQAAAAGQAA
    ABMAAAAQAAAAEAAAABkAAAATAAAAEAAAABAAAAAZAAAAEwAAABAAAAAQAAAAGQAAABMAAAAQAAAA
    EAAAABkAAAATAAAAEAAAABAAAAAZAAAAEwAAABAAAAAQAAAAGQAAABMAAAAQAAAAEAAAABkAAAAT
    AAAAEAAAABAAAAAlAAAAEwAAABAAAAAQAAAAPAAAAFQAAACWAAAAQwAAAMMAAABKAAAAQgAAAHgA
    AABsAAAAQwAAABsAAAAXAAAAEwAAACoAAAAXAAAAEgAAABIAAAAcAAAAFQAAABMAAAASAAAAGQAA
    ABMAAAAQAAAAEAAAABkAAAATAAAAEAAAABAAAAAZAAAAEwAAABAAAAAQAAAAGAAAABMAAAAQAAAA
    EAAAABgAAAASAAAAFHN0Y28AAAAAAAAAAQAAACwAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxy
    AAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAA
    TGF2ZjU3LjgzLjEwMA==
    ">
      Your browser does not support the video tag.
    </video>


