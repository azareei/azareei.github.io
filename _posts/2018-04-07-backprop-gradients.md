---
layout: post
title: Backpropagation in Neural Networks
date: 2018-04-07 16:34:00
---

<h2> Backpropagation of Gradients </h2>

Layer \( l\) values in the neural network after applying activation function is stored in a column vecotr \( {a}^l\). The subscript represent the layer number. The connections are stored in a weight matrix \( W^l \), and the bias column vector is assumed to be \( b^l\). The forward propagation is then obtained as

</p>

<p>
\[ a^l = \sigma ( W^l a^{l-1} + b^l) \]
</p>

<p>
We introduce a new vector  \( z^l = W^l a^{l-1} + b^l \), which is the layer \( l \) values before applying the transfer function \( \sigma() \). In summary, the neural network procedure is summarized as  
\[ a^l = \sigma ( z^l ) \]
\[ z^l = W^l a^{l-1} + b^l \]
</p>

<p>
To simplify the procedure, we assume a simple 3 layer network as 
</p>

<div class="row mt-3" style="text-align:center;">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" width="500" src="{{site.baseurl}}/assets/posts/network-sample-3layer.png">
    </div>
</div>

<p>
In this network, we have  
</p>
<ul class="org-ul">
<li>Input:  \( a^0 \)</li>
<li>Layer1: \( z^1 = W^1 a^0 + b^1 \) and \(a^1 = \sigma (z^1) \)</li>
<li>Layer2: \( z^2 = W^2 a^1 + b^2 \) and \(a^2 = \sigma (z^2) \)</li>
</ul>

<p>
where the dimensions are \( [W^1]_{ 4 \times 3} \), \( [W^2]_{ 2
\times 4}\), also \([a^0]_{ 3\times 1}\), \( [a^1] _{4 \times 1},
[b^1] _{4 \times 1}, [z^1] _{4 \times 1}\), and \( [z^2]_{2\times 1},
[a^2]_{2\times 1} \).  Our objective is minimize the distance between
the network's output \( a^2\) and \( t\). A simple cost function here
would be \( C = \frac{1}{2} || a^2 - t ||^2 \). The weights matrix and
bias vectors are updated usinng gradient descent method, where we need to find the derivatives of the loss
function with respect to the weights and biases as 
</p>

<p>
\[ W^l = W^l - \alpha \frac{ \partial C}{\partial W^l}, \quad b^l = b^l - \alpha \frac{ \partial C}{\partial b^l}, \qquad \qquad l=\{1,2\} \] 
</p>

<p>
We start by finding the derivatives of the weights w.r.t \( W^2 \), we have
\[ \frac{\partial C}{\partial W^2} = ( a^2 - t) \frac{\partial a^2}{\partial W^2} = (a^2-t) \circ \sigma'(z^2) \frac{\partial z^2}{\partial W^2} = (a^2-t) \circ \sigma'(z^2) \frac{\partial W^2 a^1 + b^2}{\partial W^2} \]
where \( A\circ B\) is the entry-wise product (i.e. \( (A\circ B)_i = A_i B_i\)). We further simplify the above relation as 
\[ \frac{\partial C}{\partial W^2}=  \left[ (a^2-t) \circ \sigma '(z^2) \right] [a^1]^\top   = \delta^2 [a^1]^\top, \qquad   \delta^2 \equiv (a^2-t) \circ  \sigma'(z^2)  \]
</p>

<p>
Notice that the dimensions work out perfectly as 
</p>

<p>
\[ \left[ \frac{ \partial C}{\partial W^2} \right] _{2\times 4} = \left[ \delta^2\right] _{2\times 1} \left[ a^1\right]_{4\times 1}^\top \]
</p>

<p>
Furthermore, we need to calculate the derivatives with respect to \( b^2 \), where we similarly find that 
</p>

<p>
\[ \left[ \frac{ \partial C}{\partial b^2} \right] _{2\times 1} = (a^2-t) \circ \sigma'(z^2) \frac{\partial W^2 a^1 + b^2}{\partial b^2} = \left[ \delta^3\right] _{2\times 1} \]
</p>

<p>
Now, we take the derivatives with respect to \( W^1 \), we find
</p>

<p>
\[ \displaylines{\frac{\partial C}{\partial W^1} = ( a^2 - t) \frac{\partial a^2}{\partial W^1} = \left[ (a^2-t) \circ \sigma'(z^2)\right]  \frac{\partial z^2}{\partial W^1} = \delta^2 \frac{\partial W^2 a^1 + b^2}{\partial W^1} \\ \\ =  [W^2]^\top \delta^3 \circ \sigma'(z^1) \frac{\partial z^1}{\partial W^1} =  [W^2]^\top \delta^3 \circ \sigma'(z^1) \frac{\partial W^1 a^0 + b^1 }{\partial W^1} }\]
</p>

<p>
To summarize we have 
</p>

<p>
\[ \frac{\partial C}{\partial W^1} = \left[ (W^2)^\top \delta^3 \circ \sigma'(z^1)\right]  [ a^0]^\top  = \delta^1 [a^0] ^\top, \qquad \delta ^1 [\equiv W^2]^\top \delta^2 \circ \sigma'(z^1)\]
</p>

<p>
In terms of dimensions we have
</p>

<p>
\[ \left[ \frac{\partial C}{\partial W^1}\right]_{4\times 3} =  \left[ \delta^1\right]_{4\times 1} \left[ a^0 \right]_{3\times 1}^\top \qquad \left[\delta ^1\right]_{4\times 1} = \left[W^2_{2\times 4} \right]^\top \left[\delta^3\right]_{2\times 1} \circ \left[\sigma'(z^1)\right]_{4\times 1} \]
</p>

<p>
Similarly taking the derivatives w.r.t the bias vector \( b^1 \), we find that  
</p>

<p>
\[ \left[ \frac{\partial C}{\partial b^1}\right]_{4\times 1} = \left[ \delta^1\right]_{4\times 1} \]
</p>

<p>
To summarize, we obtained the following result  
</p>

<p>
\[ \frac{ \partial C}{\partial W^2} =  \delta^2 [a^1]^\top, \qquad \frac{ \partial C}{\partial b^2} =  \delta^2, \qquad \delta^2 \equiv  (a^2-t) \circ  \sigma'(z^2) \] 
</p>

<p>
\[ \frac{\partial C}{\partial W^1} = \delta^1 [a^0] ^\top, \qquad \frac{\partial C}{\partial b^1} = \delta^1, \qquad \delta ^1 \equiv [W^2]^\top \delta^3 \circ \sigma'(z^1)  \] 
</p>

<p>
Note that if we increase the number of layers, a similar pattern shows up. 
</p>
