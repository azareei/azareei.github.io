---
layout: posts-article
title: Neural Network for a Nonlinear Function 
category: machine-learning
---

<br>


<h2> Learning a nonlinear function using neural network! </h2>


<p>
Here we show how to create a neural net to learn a nonlinear function using <a href="https://github.com/HIPS/autograd">autograd</a> package for calculating gradients. For the nonlinear function, I choose \( f(x) = x^3 - 2*x^22 - 5x + 6 \) on the domain \( [ -2,4] \).  Here is the code:
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

def neural_net_output(params, inputs):
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
x = np.linspace(-2,4,100).reshape((-1,1));

def nonlinear_function(x):
    """" returns a nonlinear function of x"""
    return x**3 - 2.0*x**2 - 5.0*x + 6


def objective_function(params, step):
    """ the output of the function should be the same
        as the nonlinear function defined in y"""
    return np.mean((nonlinear_function(x) - neural_net_output(params, x))**2)


def callback(params, step, g):
    if step%1000 ==0:
        print("At itteration {0:4d} objective value is: {1}".format(step, objective_function(params,step)));


params = adam(grad(objective_function), params, step_size=0.01, num_iters=10000, callback=callback);


# plotting the final result
import matplotlib.pyplot as plt
plt.plot(x, neural_net_output(params, x),'r')
plt.plot(x, nonlinear_function(x),'b')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.xlim([-3, 5])
plt.savefig('neural-net-function.png')
</pre>
</div>



<div class="figure">
<p><img src="{{site.baseurl}}/img/machine_learning/neural-net-function.png" alt="neural-net-function.png" />
</p>
</div>

<p>
So we modeled a nonlinear function using a neural network which worked well. With only 10,000 iterations we reached \( 10^{-4} \) accuracy which is impressive to me. 
</p>
