#models.py
#Akshay Krishna

# all these functions must have a signature
# def fun(x, y, **kwargs)
# and must return an array of the same dimension as y
# The first element of y should always be the output/Neuron state variable
import numpy as np


def Yamada_1(x, y, a=2., A=6.5, B=-6., gamma1=1, gamma2=1, kappa=50, beta=2e-1):
    """
    Full yamada model with input into gain medium.

    :math:`y=(I, G, Q)`, where I is the laser field intensity, and G and Q are gain and 
    saturable absorber inversions. :math:`x=i_{in}` is input current, A and B are gain and absorber 
    pump rates, a is the medium gain ratio, :math:`\\gamma_i`, are material decay rates, 
    :math:`\\kappa` is cavity loss rate, and :math:`\\beta` is photon noise:

    .. math:: 
        \\dot{I} &= -\\kappa(1-G-Q)I+\\beta 
        
        \\dot{G} &= \\gamma_1(A-G-IG)+i_{in}(t)

        \\dot{Q} &= \\gamma_2(B-Q-aIQ) 
    """
    return np.array([-kappa*(1-y[1]-y[2])*y[0]+beta,
                gamma1*(A-(1+y[0])*y[1])+x,
                gamma2*(B-(1+a*y[0])*y[2]) ])

def Yamada_0(x, y, P=0.8, gamma=1, kappa=50, beta=5e-1):
    """
    A simplification to **Yamada_1** with  a single medium.
    

    :math:`y=(I, J)`, with J the inversion of the gain medium 
    (:math:`J=G+Q,\\gamma_1=\\gamma_2=\\gamma, a=1`) with pump rate :math:`P=A+B` 
    and decay rate :math:`\\gamma`.  :math:`x=i_{in}` is input current, 
    :math:`\\kappa` is cavity loss rate and :math:`\\beta` is photon noise:

    .. math:: 
        \\dot{I} &= -\\kappa(1-J)I+\\beta 
        
        \\dot{J} &= \\gamma(P-J-IJ)+i_{in}(t)
    """
    return np.array([-kappa*(1-y[1])*y[0]+beta,
                gamma*(P-(1+y[0])*y[1])+x ])

def Yamada_2(x, y, a=1., A=6.5, B=-6., gamma1=1, gamma2=1, kappa=50, beta=2e-1):
    """
     **Yamada_1** with input to cavity directly instead of gain medium.

    .. math:: 
        \\dot{I} &= -\\kappa(1-G-Q)I+\\beta )+i_{in}(t)
        
        \\dot{G} &= \\gamma_1(A-G-IG)

        \\dot{Q} &= \\gamma_2(B-Q-aIQ) 


    """
    return np.array([-kappa*(1-y[1]-y[2])*y[0]+beta+x,
                gamma1*(A-(1+y[0])*y[1]),
                gamma2*(B-(1+a*y[0])*y[2]) ])

def FitzHughNagumo(x, y, a=0.7, b=0.8, tau=12.5):
    """
    FitzHugh-Nagumo neuron model

    A simplification of the Hodgekin-Huxley model of a squid Neuron.
    :math:`y=(V, W)` where  V is the membrane potential and 
    W the ion conductivity, a slower recovery variable.  :math:`x` is the 
    magnitude of the stimulus current and :math:`\\{a,b,\\tau\\}` are fit parameters:

    .. math:: 
        \\dot{V} &= V - V^3/3 - W + x(t)
        
        \\dot{W} &= (V+a-bW)/\\tau
    """
    return np.array([y[0] - y[0]**3/3.0 - y[1] + x,
                     (y[0] + a - b*y[1]) / tau ])

def identity(x, y, h):
    """
    The identity neuron, output=input

    A neuron which returns the input at the previous timestep,
    i.e. :math:`y(n+1)=x(n)`.  h is the ODE solver timestep and we thus
    have the ODE:

    .. math::
        \\dot{y} = (x-y)/h
    """
    return 1.0*(x-y) / h
