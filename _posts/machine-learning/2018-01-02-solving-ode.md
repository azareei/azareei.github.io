---
layout: posts-article
title: Neural Network for a Nonlinear Function 
category: machine-learning
---

<br>


<h2> Learning a nonlinear function using neural network! </h2>


<p>
Here, I want to solve an ode using a neural network. I consider the
most basic ordinary differential equation as 
</p>

<p>
\[ \frac{d y}{d x} = -\alpha y, \quad y(x=x_0) = y_0\]
</p>


<p>
Lets use neural network to solve this ODE. The neural net will be
similar to the learning of the nonlinear function in the previous
post. Again we use <a href="https://github.com/HIPS/autograd">autogra</a> python library for calculating derivatives.
The code is as follows
</p>

<div class="org-src-container">
<pre class="src src-python">import autograd.numpy as np
from autograd import grad, elementwise_grad
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr


def init_random_parameters(scale, layer_sizes, rs = npr.RandomState(0)):
    """ Building a list of tuples (weights, biases)
        For each layer, defined in layer_sizes
        and saceling with scale parameteR"""
    return [ (scale*rs.randn(m,n),  #weighths in matrix of m input and n output
              scale*rs.randn(n))    #biases
             for m,n in zip(layer_sizes[:-1], # m: number of input layers
                            layer_sizes[1:])] # n: number of output layers


def logistic(x):
    """ defining a logistic term"""
    return 1.0 / (1.0 + np.exp(-x));

def f(params, inputs):
    """This function calculates the output of the neural net
        with a list of (weights, bias) tuples as params.
        returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = logistic(outputs)
    return outputs

# initializing the neural net with random numbers
params = init_random_parameters(0.5, layer_sizes = [1,5,5,1]);

# defining the domain
t = np.linspace(0,4,100).reshape((-1,1));

# defining the derivative of the function

dfdt = elementwise_grad(f,1); 
alpha = 1.0;
f0 = 10.0;

def f_exact(t):
    return f0*np.exp(-alpha*t);

def objective_function(params, step):
    """ the output of the function should be the same
        as the nonlinear function defined in y"""
    ode = dfdt(params,t) - (-alpha*f(params,t))
    initial_condition = f(params,0);
    return np.mean(ode**2) + (initial_condition - f0)**2


def callback(params, step, g):
    if step%1000 ==0:
        print("At itteration {0:4d} objective value is: {1}".format(step, objective_function(params,step)));


params = adam(grad(objective_function), params, step_size=0.01, num_iters=10000, callback=callback);


# plotting the final result
import matplotlib.pyplot as plt
plt.plot(t, f(params, t),'r')
plt.plot(t, f_exact(t),'b')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.savefig('ode-neural-net.png')

</pre>
</div>



<div class="figure">
<p><img src="{{site.baseurl}}/img/machine_learning/ode-neural-net.png" alt="ode-neural-net.png" />
</p>
</div>



<p>
As we can see with the accuracy of \( 10^{-5} \), the solution matches
with the exact solution.
</p>



<p>
A usual way to sove such ordinary differential equations is to use ode45 built in function in python. Here is how we can use scipy integrator to solve such ode: 
</p>

<div class="org-src-container">
<pre class="src src-python"># integrating using ode45
from scipy.integrate import ode
import numpy as numpy

def f(t,y,arg1):
    return -arg1*y
y0, t0 = 10.0, 0.0;
alpha = 1.0; # parameter in dy/dt = alpha y

r = ode(f).set_integrator('dopri5', verbosity = 1);
r.set_initial_value([y0],t0).set_f_params(alpha);

tf = 4;
dt= 0.05;
num_steps = numpy.int((tf-t0)/dt) + 1;


t = numpy.zeros((num_steps,1))
y = numpy.zeros((num_steps,1));


k = 0;
while r.successful() and r.t &lt; tf:
    r.integrate(r.t + dt)
    print("{0}{1}".format(r.t, r.y))
    t[k] = r.t;
    y[k] = r.y;
    k = k+ 1;

import matplotlib.pyplot as plt

plt.plot(t, y,'r');
plt.plot(t,y0*numpy.exp(-alpha*t),'b')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.savefig('ode-45.png')
</pre>
</div>


<div class="figure">
<p><img src="{{site.baseurl}}/img/machine_learning/ode-45.png" alt="ode-45.png" />
</p>
</div>
