---
layout: posts-article
title: Solving BVP using neural net 
category: machine-learning
---

<br>


<h2> How to solve a BVP using neural net? </h2>



<p>
Here we will use a neural network to solve a boundary value problem. For the boundary value equation we choose \( y'' + y = 0\), with the boundary conditions of \( y(0) = 1\) and \( y(\pi/2)=1\). This equation has a unique solution given by \( y(x) = \sin(x) + \cos(x) \). The problem set is as before, however we need to choose the objective function slightly different. Note thta BVPs can have infinite number of solutions, here we chose a BVP that has a unique solution. Here is the python code
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
<span style="color: #fcaf3e;">x</span> = np.linspace(0,np.pi/2,100).reshape((-1,1));

<span style="color: #5F7F5F;"># </span><span style="color: #73d216;">defining the derivative of the function</span>

<span style="color: #fcaf3e;">fp</span> = elementwise_grad(f,1);
<span style="color: #fcaf3e;">fpp</span> = elementwise_grad(fp,1); 

<span style="color: #fcaf3e;">f0</span> = 1.0;
<span style="color: #fcaf3e;">fpi2</span> = 1.0;


<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">f_exact</span>(x):
    <span style="color: #b4fa70; font-weight: bold;">return</span> np.sin(x) + np.cos(x);

<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">objective_function</span>(params, step):
    <span style="color: #9FC59F;">""" the output of the function should be the same</span>
<span style="color: #9FC59F;">        as the nonlinear function defined in y"""</span>
    <span style="color: #fcaf3e;">bvp</span> = fpp(params,x) + f(params,x)
    <span style="color: #fcaf3e;">bc1</span> = f(params,0) - f0;
    <span style="color: #fcaf3e;">bc1</span> = f(params,np.pi/2.0) - fpi2;    
    <span style="color: #b4fa70; font-weight: bold;">return</span> np.mean(bvp**2) + bc1**2 + bc2**2


<span style="color: #b4fa70; font-weight: bold;">def</span> <span style="color: #fce94f;">callback</span>(params, step, g):
    <span style="color: #b4fa70; font-weight: bold;">if</span> step%1000 ==0:
        <span style="color: #b4fa70; font-weight: bold;">print</span>(<span style="color: #e9b96e;">"At itteration {0:4d} objective value is: {1}"</span>.<span style="color: #e090d7; font-weight: bold;">format</span>(step, objective_function(params,step)));


<span style="color: #fcaf3e;">params</span> = adam(grad(objective_function), params, step_size=0.01, num_iters=10000, callback=callback);


<span style="color: #5F7F5F;"># </span><span style="color: #73d216;">plotting the final result</span>
<span style="color: #b4fa70; font-weight: bold;">import</span> matplotlib.pyplot <span style="color: #b4fa70; font-weight: bold;">as</span> plt
plt.plot(x, f(params, x),<span style="color: #e9b96e;">'r'</span>)
plt.plot(x, f_exact(x),<span style="color: #e9b96e;">'b'</span>)
plt.xlabel(<span style="color: #e9b96e;">'$x$'</span>)
plt.ylabel(<span style="color: #e9b96e;">'$f(x)$'</span>)
plt.savefig(<span style="color: #e9b96e;">'bvp-neural-net.png'</span>)

</pre>
</div>



<div class="figure">
<p><img src="{{site.baseurl}}/img/machine_learning/bvp-neural-net.png" alt="bvp-neural-net.png" />
</p>
</div>

<p>
Again we got the exact solution with accuracy of \( 10^{-6}\) with 10,000 iterations. 
</p>
