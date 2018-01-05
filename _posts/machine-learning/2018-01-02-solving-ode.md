---
layout: posts-article
title: Solving ODE using neural net 
category: machine-learning
---

<br>


<h2> How to solve an ODE using neural net? </h2>




<p>
Here, I want to solve an ode using a neural network. I consider the
most basic ordinary differential equation as 
</p>

<p>
\[ \frac{d y}{d x} = \alpha y, \quad y(x=x_0) = y_0\]
</p>


<p>
Lets use neural network to solve this ODE. The neural net will be
similar to the learning of the nonlinear function in the previous
post. Again we use <a href="https://github.com/HIPS/autograd">autogra</a> python library for calculating derivatives.
The code is as follows
</p>

<div class="org-src-container">
<pre class="src src-python"><span style="color: #b4fa70; font-weight: bold;">import</span> autograd.numpy <span style="color: #b4fa70; font-weight: bold;">as</span> np
<span style="color: #b4fa70; font-weight: bold;">from</span> autograd <span style="color: #b4fa70; font-weight: bold;">import</span> grad, elementwise_grad
<span style="color: #b4fa70; font-weight: bold;">from</span> autograd.misc.optimizers <span style="color: #b4fa70; font-weight: bold;">import</span> adam
<span style="color: #b4fa70; font-weight: bold;">import</span> autograd.numpy.random <span style="color: #b4fa70; font-weight: bold;">as</span> npr


<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">init_random_parameters</span>(scale, layer_sizes, rs = npr.RandomState(0)):
    <span style="color: #9FC59F;">""" Building a list of tuples (weights, biases)</span>
<span style="color: #9FC59F;">        For each layer, defined in layer_sizes</span>
<span style="color: #9FC59F;">        and saceling with scale parameteR"""</span>
    <span style="color: #b4fa70; font-weight: bold;">return</span> [ (scale*rs.randn(m,n),  <span style="color: #5F7F5F;">#</span><span style="color: #73d216;">weighths in matrix of m input and n output</span>
              scale*rs.randn(n))    <span style="color: #5F7F5F;">#</span><span style="color: #73d216;">biases</span>
             <span style="color: #b4fa70; font-weight: bold;">for</span> m,n <span style="color: #b4fa70; font-weight: bold;">in</span> <span style="color: #e090d7; font-weight: bold;">zip</span>(layer_sizes[:-1], <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">m: number of input layers</span>
                            layer_sizes[1:])] <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">n: number of output layers</span>


<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">logistic</span>(x):
    <span style="color: #9FC59F;">""" defining a logistic term"""</span>
    <span style="color: #b4fa70; font-weight: bold;">return</span> 1.0 / (1.0 + np.exp(-x));

<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">f</span>(params, inputs):
    <span style="color: #9FC59F;">"""This function calculates the output of the neural net</span>
<span style="color: #9FC59F;">        with a list of (weights, bias) tuples as params.</span>
<span style="color: #9FC59F;">        returns normalized class log-probabilities."""</span>
    <span style="color: #b4fa70; font-weight: bold;">for</span> W, b <span style="color: #b4fa70; font-weight: bold;">in</span> params:
        <span style="color: #fcaf3e;">outputs</span> = np.dot(inputs, W) + b
        <span style="color: #fcaf3e;">inputs</span> = logistic(outputs)
    <span style="color: #b4fa70; font-weight: bold;">return</span> outputs

<span style="color: #5F7F5F;"># </span><span style="color: #73d216;">initializing the neural net with random numbers</span>
<span style="color: #fcaf3e;">params</span> = init_random_parameters(0.5, layer_sizes = [1,5,5,1]);

<span style="color: #5F7F5F;"># </span><span style="color: #73d216;">defining the domain</span>
<span style="color: #fcaf3e;">t</span> = np.linspace(0,4,100).reshape((-1,1));

<span style="color: #5F7F5F;"># </span><span style="color: #73d216;">defining the derivative of the function</span>

<span style="color: #fcaf3e;">dfdt</span> = elementwise_grad(f,1); 
<span style="color: #fcaf3e;">alpha</span> = 1.0;
<span style="color: #fcaf3e;">f0</span> = 10.0;

<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">f_exact</span>(t):
    <span style="color: #b4fa70; font-weight: bold;">return</span> f0*np.exp(-alpha*t);

<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">objective_function</span>(params, step):
    <span style="color: #9FC59F;">""" the output of the function should be the same</span>
<span style="color: #9FC59F;">        as the nonlinear function defined in y"""</span>
    <span style="color: #fcaf3e;">ode</span> = dfdt(params,t) - (-alpha*f(params,t))
    <span style="color: #fcaf3e;">initial_condition</span> = f(params,0);
    <span style="color: #b4fa70; font-weight: bold;">return</span> np.mean(ode**2) + (initial_condition - f0)**2


<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">callback</span>(params, step, g):
    <span style="color: #b4fa70; font-weight: bold;">if</span> step%1000 ==0:
        <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">"At itteration {0:4d} objective value is: {1}"</span>.<span style="color: #e090d7; font-weight: bold;">format</span>(step, objective_function(params,step)));


<span style="color: #fcaf3e;">params</span> = adam(grad(objective_function), params, step_size=0.01, num_iters=10000, callback=callback);


<span style="color: #5F7F5F;"># </span><span style="color: #73d216;">plotting the final result</span>
<span style="color: #b4fa70; font-weight: bold;">import</span> matplotlib.pyplot <span style="color: #b4fa70; font-weight: bold;">as</span> plt
plt.plot(t, f(params, t),<span style="color: #e9b96e;">'r'</span>)
plt.plot(t, f_exact(t),<span style="color: #e9b96e;">'b'</span>)
plt.xlabel(<span style="color: #e9b96e;">'$t$'</span>)
plt.ylabel(<span style="color: #e9b96e;">'$f(t)$'</span>)
plt.savefig(<span style="color: #e9b96e;">'ode-neural-net.png'</span>)

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
<pre class="src src-python"><span style="color: #5F7F5F;"># </span><span style="color: #73d216;">integrating using ode45</span>
<span style="color: #b4fa70; font-weight: bold;">from</span> scipy.integrate <span style="color: #b4fa70; font-weight: bold;">import</span> ode
<span style="color: #b4fa70; font-weight: bold;">import</span> numpy <span style="color: #b4fa70; font-weight: bold;">as</span> numpy

<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">f</span>(t,y,arg1):
    <span style="color: #b4fa70; font-weight: bold;">return</span> -arg1*y
<span style="color: #fcaf3e;">y0</span>, <span style="color: #fcaf3e;">t0</span> = 10.0, 0.0;
<span style="color: #fcaf3e;">alpha</span> = 1.0; <span style="color: #5F7F5F;"># </span><span style="color: #73d216;">parameter in dy/dt = alpha y</span>

<span style="color: #fcaf3e;">r</span> = ode(f).set_integrator(<span style="color: #e9b96e;">'dopri5'</span>, verbosity = 1);
r.set_initial_value([y0],t0).set_f_params(alpha);

<span style="color: #fcaf3e;">tf</span> = 4;
<span style="color: #fcaf3e;">dt</span>= 0.05;
<span style="color: #fcaf3e;">num_steps</span> = numpy.<span style="color: #e090d7; font-weight: bold;">int</span>((tf-t0)/dt) + 1;


<span style="color: #fcaf3e;">t</span> = numpy.zeros((num_steps,1))
<span style="color: #fcaf3e;">y</span> = numpy.zeros((num_steps,1));


<span style="color: #fcaf3e;">k</span> = 0;
<span style="color: #b4fa70; font-weight: bold;">while</span> r.successful() <span style="color: #b4fa70; font-weight: bold;">and</span> r.t &lt; tf:
    r.integrate(r.t + dt)
    <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">"{0}{1}"</span>.<span style="color: #e090d7; font-weight: bold;">format</span>(r.t, r.y))
    <span style="color: #fcaf3e;">t</span>[k] = r.t;
    <span style="color: #fcaf3e;">y</span>[k] = r.y;
    <span style="color: #fcaf3e;">k</span> = k+ 1;

<span style="color: #b4fa70; font-weight: bold;">import</span> matplotlib.pyplot <span style="color: #b4fa70; font-weight: bold;">as</span> plt

plt.plot(t, y,<span style="color: #e9b96e;">'r'</span>);
plt.plot(t,y0*numpy.exp(-alpha*t),<span style="color: #e9b96e;">'b'</span>)
plt.xlabel(<span style="color: #e9b96e;">'$x$'</span>)
plt.ylabel(<span style="color: #e9b96e;">'$f(x)$'</span>)
plt.savefig(<span style="color: #e9b96e;">'ode-45.png'</span>)
</pre>
</div>


<div class="figure">
<p><img src="{{site.baseurl}}/img/machine_learning/ode-45.png" alt="ode-45.png" />
</p>
</div>
