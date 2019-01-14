#models.py
#Akshay Krishna

# all these functions must have a signature
# def fun(x, y, **kwargs)
# and must return an array of the same dimension as y
# The first element of y should always be the output/Neuron state variable
import numpy as np

def Yamada_0(x, y, P=0.8, gamma=1, kappa=50, beta=5e-1):
    """
    Simplified Yamada model with gamma1=gamma2=gamma, a=1, J=G+Q, 
    y=(I, J), x is input current, P=A+B is Pump rates, 
    gamma is material decay rate, kappa is cavity loss rate
    beta is photon noise
    """
    return np.array([-kappa*(1-y[1])*y[0]+beta,
                gamma*(P-(1+y[0])*y[1])+x ])


def Yamada_1(x, y, a=2., A=6.5, B=-6., gamma1=1, gamma2=1, kappa=50, beta=2e-1):
    """
    Full yamada model with input into gain medium.

    .. math:: 'y=(I, G, Q)', where I is the laser field intensity, and G and Q are gain and 
    saturable absorber inversions.  .. math::'x=i_{in}' is input current, A and B are gain and absorber 
    pump rates, a is the medium gain ratio, .. math::'\gamma_i' are material decay rates, .. math::'\kappa' is cavity loss rate, and
    .. math::'\beta' is photon noise:
    .. math::`\begin{equation} \dot{I}=(1-G-Q)I+\beta \\
    \dot{G}=\gamma_1(A+i_{in}(t)-G-IG) \\
    \dot{Q}=\gamma_2(B-Q-aIQ) \end{equation}`
    """
    return np.array([-kappa*(1-y[1]-y[2])*y[0]+beta,
                gamma1*(A-(1+y[0])*y[1])+x,
                gamma2*(B-(1+a*y[0])*y[2]) ])

def Yamada_2(x, y, a=1., A=6.5, B=-6., gamma1=1, gamma2=1, kappa=50, beta=2e-1):
    """
    full yamada model with direct cavity input
    y=(I, G, Q), x is input current, A and B are Pump rates, 
    gamma are material decay rates, kappa is cavity loss rate
    beta is photon noise
    """
    return np.array([-kappa*(1-y[1]-y[2])*y[0]+beta+x,
                gamma1*(A-(1+y[0])*y[1]),
                gamma2*(B-(1+a*y[0])*y[2]) ])

def FitzHughNagumo(x, y, a=1.0, b=1.0, tau=1.0):
    """
    The y's are phase space variables (outputs), the x is the scalar input
    d y1 / dt = y1 - y1^3/3 - y2 + x
    d y2 / dt = (y1 + a - b*y2) / tau
    parameters: a, b, tau
    """
    return np.array([y[0] - y[0]**3/3.0 - y[1] + x,
                     (y[0] + a - b*y[1]) / tau ])

def identity(x, y, h):
    """
    should return output = input
    y_n+1 = x_n, or alternatively
    dy / dt = (x-y) / h
    where h is the Euler time step
    """
    return 1.0*(x-y) / h
