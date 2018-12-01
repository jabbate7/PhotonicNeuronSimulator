def Yamada(x, beta):
    #define function equations
    pass

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
    return (x-y) / h