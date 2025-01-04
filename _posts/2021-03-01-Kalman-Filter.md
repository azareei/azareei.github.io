---
layout: post
title:  Kalman Filter
date: 2021-03-01 22:10:00-0400
description: Kalman Filter Tutorial
---


Kalman filter is a classic state estimation technique that has found application in many places. In this simple tutorial, I will try to explain Kalman filter in an intuitive way. This is the most basic introduction to the Kalman filter and basically how I learned it. Before getting to the Kalman filter, I will first review some basic materials that we need. 

<br>
# Prerequisite

Let $$x_i$$ be a random variable that has a **probability density function** $$p_i(x)$$ whose mean and variance are $$\mu_i$$ and $$\sigma_i^2$$. We write $$x_i \sim p_i(\mu_i,\sigma_i^2)$$. 

Assuming a set of pairwise uncorrelated random variables $$x_1 \sim p_1(\mu_1,\sigma_1^2), \cdots x_n \sim p_n(\mu_n,\sigma_n^2)$$, if $$y$$ is a random variable where $$y = \sum_{i=1}^n \alpha_i x_i$$, then the mean and variance of $$y$$ are 

$$\mu_y = \sum_{i=1}^n \alpha_i \mu_i$$

$$\sigma_y^2 = \sum_{i=1}^n \alpha_i \sigma_i^2$$

<br>
## Fusing two variables

Now, imagine that we want to measure a variable $$y$$, we have two totally different devices where they use different methods, one is based on an old method for example and its results are reported with $$x_1 \sim p_1(\mu_1,\sigma_1^2)$$, and one that uses a new method and its the results are reported with $$x_2 \sim p_2(\mu_2,\sigma_2^2)$$. Now the question is how to combine these two different measurements to create an optimal estimator for $$y$$.  The simplest way is to combine these results linearly as $$y = \alpha x_1 + \beta x_2$$.  A reasonable requirement is that if the two estimates $$x_1$$ and $$x_2$$ are giving the same result, then this linear combination should give out that same result. This implies that $$\alpha + \beta =1$$. So our linear estimator so far becomes

$$y_\alpha(x_1,x_2) = \alpha x_1 + (1-\alpha)x_2$$

But what value should we pick for $$\alpha$$? One reasonable way is to say that the optimal value of $$\alpha$$ minimizes the variance of $$y_\alpha$$. The variance of $$y_\alpha$$  is 

$$\sigma_y^2 = \alpha^2 \sigma_1^2 + (1-\alpha)^2 \sigma_2^2$$

$$\frac{d}{d \alpha} \sigma_y^2 = 2\alpha \sigma_1^2 -2 (1-\alpha)\sigma_2^2 = 0 \to \alpha = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

Since the second derivative is positive then this value of $$\alpha$$ minimizes the variance. The estimator then becomes

$$y(x_1,x_2) = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2} x_1 + \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} x_2$$

$$y(x_1,x_2) = \frac{1/\sigma_1^2}{1/\sigma_1^2 + 1/\sigma_2^2} x_1 + \frac{1/\sigma_2^2}{1/\sigma_1^2 + 1/\sigma_2^2} x_2, \quad \sigma_y^2 = \frac{1}{1/\sigma_2^1 + 1/\sigma_2^2}$$

<br>
## Fusing multiple variables

The above argument can be extended for multiple scalar estimates. Let $$x_i \sim p_i(\mu_i,\sigma_i^2)$$ be a set of pairwise uncorrelated random variables. Consider unbiased linear estimator $$y = \sum_{i=1}^n \alpha_i x_i$$.  Using Lagrange multipliers, we have

$$f(\alpha_1, \cdots, \alpha_n) = \sum_{i=1}^n \alpha_i^2 \sigma_i^2 + \lambda \left( \sum_{i=1}^n \alpha_i -1 \right)$$

where $$\lambda$$  is the Lagrange multiplier. Taking the derivative with respect to $$\alpha_j$$ we find that $$\alpha_1 \sigma_1^2 = \alpha_2\sigma_2^2 = \cdots = -\lambda/2$$. Since $$\sum \alpha_i = 1$$, then we can find that 

$$\alpha_i = \frac{\frac{1}{\sigma_i^2}}{\sum_{i=1}^n \frac{1}{\sigma_i^2}}$$

where the variance $$\sigma_y$$  is 

$$\sigma_y = \frac{1}{\sum_{i=1}^n \frac{1}{\sigma_i^2}}$$

<br>
## Vector estimates

Now let's expand the same result to the vectors of random variables. Let $$\mathbf{x}_1 \sim p_1( \mathbf{\mu}_1,\Sigma_1), \cdots, \mathbf{x}_n \sim p_n( \mathbf{\mu}_n,\Sigma_n)$$ be a set of pairwise uncorrelated random variables of length $$m$$. If random variable $$\mathbf{y}$$ is a linear combination of these random variables as $$\mathbf{y}  = \sum_{i=1}^n \mathbf{A}_i \mathbf{x}_i$$, then the mean and covariance of $$\mathbf{y}$$ is obtianed as 

$$\mathbf{\mu}_\mathbf{y} = \sum_{i=1}^n \mathbf{A}_i \mathbf{\mu}_i$$

$$\Sigma_\mathbf{yy} = \sum_{i=1}^n \mathbf{A}_i \Sigma_i\mathbf{A}^\top_i$$

<br>
## Fusing multiple vector estimates

Imagine the linear estimator as 

$$\mathbf{y}(\mathbf{x}_1,\cdots,\mathbf{x}_n) = \sum_{i=1}^n \mathbf{A}_i \mathbf{x}_i, \quad \sum \mathbf{A}_i = \mathbb{I}$$

Similarly, we intend to minimize $$\mathbb{E}[ (\mathbf{y}-\mu)^\top (\mathbf{y}-\mu)]$$ . We define the following optimization problem using Lagrangian multipliers

$$f(\mathbf{A}_1, \cdots, \mathbf{A}_n) = \mathbb{E} \left[\sum_{i=1}^n (\mathbf{x}_i-\mu_i)^\top \mathbf{A}^\top_i \mathbf{A}_i (\mathbf{x}_i - \mu_i) \right] + \langle \Lambda, \mathbf{A}_i-\mathbb{I}\rangle$$

where the second term is the Lagrangian multipliers and $$\langle \Lambda, \mathbf{A}_i-\mathbb{I}\rangle = \text{tr}\left[\Lambda^\top\left(  \mathbf{A}_i-\mathbb{I}\right)\right]$$. Taking derivative of $$f$$ with respect to $$\mathbf{A}_i$$ and setting each derivative to zero to find the optimal values of $$\mathbf{A}_i$$ gives us 

$$\mathbb{E} \left[2\mathbf{A}_i (\mathbf{x}_i-\mu_i) (\mathbf{x}_i - \mu_i)^\top  +  \Lambda \right]=0$$

$$2\mathbf{A}_i \Sigma_i + \Lambda = 0\to \mathbf{A}_1 \Sigma_1 = \mathbf{A}_2 \Sigma_2  = \cdots = \mathbf{A}_n \Sigma_n  = \frac{-\Lambda}{2}  $$

Using the fact that $$\sum \mathbf{A}_i = \mathbb{I}$$, 

$$\mathbf{A}_i = \left( \sum_{i=1}^n \Sigma_j^{-1}\right)^{-1} \Sigma_i^{-1}$$

Therefore the optimal estimator becomes

$$\mathbf{y} =  \left( \sum_{i=1}^n \Sigma_j^{-1}\right)^{-1}\sum_{i=1}^n  \Sigma_i^{-1} \mathbf{x}_i, \qquad \Sigma_{\mathbf{y}\mathbf{y}} = \left( \sum_{i=1}^n \Sigma_j^{-1}\right)^{-1}$$

<br>
## Special case of $$n=2$$

Let $$\mathbf{x}_1 \sim p_1(\mu_1, \Sigma_1)$$, and $$\mathbf{x}_2 \sim p_2 (\mu_2,\Sigma_2)$$, then we have 

$$\mathbf{K} = \Sigma_1 \left(\Sigma_1 + \Sigma_2 \right)^{-1}$$

$$\mathbf{y} = \mathbf{x}_1 + \mathbf{K} (\mathbf{x}_2-\mathbf{x}_1), \quad \Sigma_{\mathbf{y}\mathbf{y}} = (\mathbf{I}-\mathbf{K})\Sigma_1$$

In order to prove the above relation, we start from the relation we obtained above, i.e.

$$\mathbf{y} = \left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1} \left( \Sigma_1^{-1} \mathbf{x}_1+ \Sigma_2^{-1}\mathbf{x}_2 \right) $$

Note that the following matrix identity holds true $$(\mathbf{A} ^{-1} + \mathbf{B}^{-1})^{-1} = \mathbf{A} (\mathbf{A} + \mathbf{B})^{-1} \mathbf{B} = \mathbf{B} (\mathbf{A} + \mathbf{B})^{-1} \mathbf{A}$$ 

$$\mathbf{y} = \Sigma_2  \left( \Sigma_1 + \Sigma_2 \right)^{-1} \Sigma_1 \Sigma_1^{-1} \mathbf{x}_1  + \Sigma_1 \left( \Sigma_1 + \Sigma_2 \right)^{-1} \Sigma_2  \Sigma_2^{-1}\mathbf{x}_2 $$

$$\mathbf{y} = \Sigma_2  \left( \Sigma_1 + \Sigma_2 \right)^{-1}  \mathbf{x}_1  + \Sigma_1 \left( \Sigma_1 + \Sigma_2 \right)^{-1} \mathbf{x}_2 $$

We add and subtract $$\Sigma_1  \left( \Sigma_1 + \Sigma_2 \right)^{-1}  \mathbf{x}_1$$  to the above equation to obtain

$$\mathbf{y} = \Sigma_2  \left( \Sigma_1 + \Sigma_2 \right)^{-1}  \mathbf{x}_1  + \Sigma_1 \left( \Sigma_1 + \Sigma_2 \right)^{-1} \mathbf{x}_2 + \Sigma_1  \left( \Sigma_1 + \Sigma_2 \right)^{-1}  \mathbf{x}_1  - \Sigma_1  \left( \Sigma_1 + \Sigma_2 \right)^{-1}  \mathbf{x}_1 $$

$$\boxed{\mathbf{y} =  \mathbf{x}_1  + \Sigma_1 \left( \Sigma_1 + \Sigma_2 \right)^{-1} \left( \mathbf{x}_2 - \mathbf{x}_1\right) }$$

Similarly for the covariance matrix we have

$$\Sigma_{\mathbf{y}\mathbf{y}} = \left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1} = \Sigma_1 \left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1}  \Sigma_2$$

We add and subtract the term $$\left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1}  \Sigma_1$$ to the above equatio to obtain

$$\Sigma_{\mathbf{y}\mathbf{y}} = \Sigma_1 \left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1}  \Sigma_2 + \Sigma_1\left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1}  \Sigma_1 - \Sigma_1\left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1}  \Sigma_1$$

$$\Sigma_{\mathbf{y}\mathbf{y}} = \Sigma_1  - \Sigma_1\left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1}  \Sigma_1$$

$$\Sigma_{\mathbf{y}\mathbf{y}} = \left( \mathbf{I}  - \Sigma_1\left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1}\right)  \Sigma_1 = \left( \mathbf{I}  - \mathbf{K} \right)  \Sigma_1 $$

<br>
## Best Linear Unbiased Estimator

Let $$\left( \begin{matrix} \mathbf{x} \\ \mathbf{y}\end{matrix}\right) \sim p\left(\left( \begin{matrix} \mu_\mathbf{x} \\ \mu_\mathbf{y} \end{matrix}\right), \left( \begin{matrix} \Sigma_\mathbf{xx}& \Sigma_\mathbf{xy} \\ \Sigma_\mathbf{yx} & \Sigma_\mathbf{yy}\end{matrix}\right) \right)$$ . The estimator $$ \hat{\mathbf{y}} = \mathbf{A}\mathbf{x} + \mathbf{b}$$ for estimating values of $$\mathbf{y}$$ for a given $$\mathbf{x}$$ is

$$\mathbf{A} = \Sigma_\mathbf{yx} \Sigma_\mathbf{xx}^{-1}$$

$$\mathbf{b} = \mu_\mathbf{y} - \mathbf{A}\mu_\mathbf{x}$$

# Kalman Filter for a linear system

Now that we know all the ingredients we can discuss the Kalman filter. Assume a linear dynamical system where 

$$\mathbf{x}_k = \mathbf{F}_k\mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k  + \mathbf{w}_k$$

where $$\mathbf{F}_k$$ is the state transition model applied to the previous state $$\mathbf{x}_{k-1},$$  and $$\mathbf{B}_k$$ is the control input model applied to the control vector $$\mathbf{u}_k$$, and $$\mathbf{w}_k$$ is the process noise assumed to be drawn from a multivariate normal distribution with $$\mathcal{N}(0,\mathbf{Q})$$ where $$\mathbf{Q}$$ is the covariance matrix. At time $$k$$, we do an observation (or measurement) $$\mathbf{z}_k$$ of the true state $$\mathbf{x}_k$$ according to the 

$$\mathbf{x}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$$

where $$\mathbf{H}_k$$ is the observation model, and $$\mathbf{v}_k$$ is the observation noise drawn from Gaussian noise $$\mathcal{N}(0,\mathbf{R}_k)$$ where $$\mathbf{R}_k$$ is the covariance matrix. 



First let's assume that $$\mathbf{H}_k= \mathbf{I}$$ where we fully observe the state. Given an estimate
that we have at time $$t-1$$ based on all the observations we had as  $$\hat{\mathbf{x}}_{t-1|t-1}$$, we make a prediction for $$\hat{\mathbf{x}}_{t|t-1}$$ based on the dynamical system equation as

$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}_t\hat{\mathbf{x}}_{t-1|t-1} + \mathbf{B}_t \mathbf{u}_t$$


Next the variance can also be estimated as 

$$\Sigma_{t|t-1} = \mathbf{F}_t \Sigma_{t-1|t-1}\mathbf{F}_t^\top + \mathbf{Q}_t$$

Given these predictions for that state at time $$t$$, we also make an observation as $$\mathbf{z}_t = \mathbf{x}_t$$ where the covariance matrix is $$\mathbf{R}_t$$.

Now our goal is to combine these results to correct our estimate of
$$\mathbf{x}_{t|t}$$
. We use the derivation that we did above to combine these results based on their covariance matrix such that the covariance is minimized, we have 


$$\boxed{\mathbf{K}_t= \Sigma_{t|t-1} \left( \Sigma_{t|t-1} + \mathbf{R}_t \right)^{-1}}$$

$$\boxed{\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \left( \mathbf{z}_t - \hat{\mathbf{x}}_{t|t-1}\right)}$$

$$\boxed{\Sigma_{t|t} = \left( \mathbf{I} - \mathbf{K}_t\right)\Sigma_{t|t-1}}$$


Now let's imagine what happens if we only do a partial observation of the state or $$\mathbf{H}_k\neq \mathbf{I}$$. In this case, we do the prediction as before, but in the step that we want to combine the results to correct the prediction, we need to make some changes since we only have partial parts of $$\mathbf{x}_t$$. In such a case, we used the best linear estimator that we introduced earlier to construct the full $$\mathbf{x}_t$$ and then use that to update the prediction. 

The estimation with partial observation becomes

$$\mathbf{H}_t\hat{\mathbf{x}}_{t|t} = \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1} + \mathbf{H}_t  \Sigma_{t|t-1} \left( \Sigma_{t|t-1} + \mathbf{R}_t \right)^{-1} \mathbf{H}_t^\top\left( \mathbf{z}_t - \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1}\right)$$

We can define

$$\boxed{\mathbf{K}_t = \Sigma_{t|t-1} \left( \Sigma_{t|t-1} + \mathbf{R}_t \right)^{-1} \mathbf{H}_t^\top}$$

and the observable simplifies to 

$$\mathbf{H}_t\hat{\mathbf{x}}_{t|t} = \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1} + \mathbf{H}_t \mathbf{K}_t\left( \mathbf{z}_t - \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1}\right)$$

The rest of the variables (hidden states) can be obtained using
$$\mathbf{C}_t\hat{\mathbf{x}}_{t|t-1}$$
where
$$\left(\begin{matrix} \mathbf{H}_t \\ \mathbf{C}_t\end{matrix} \right)$$
becomes an invertible matrix. The simplest example is to have it be equal to the identity matrix. The covariance between
$$\mathbf{C}_t\hat{\mathbf{x}}_{t|t-1}$$
and the observable
$$\mathbf{H}_t\hat{\mathbf{x}}_{t|t-1}$$
is
$$\mathbf{C}_t \Sigma_{t|t-1} \mathbf{H}_t^\top$$
. Using the best linear estimate estimator, we can find the hidden portion estimation as 

 
$$\mathbf{C}_t\hat{\mathbf{x}}_{t|t} = \mathbf{C}_t\hat{\mathbf{x}}_{t|t-1} + \left(\mathbf{C}_t \Sigma_{t|t-1} \mathbf{H}_t^\top \right) \left( \mathbf{H}_t \Sigma_{t|t-1} \mathbf{H}_t^\top  \right)^{-1} \mathbf{H}_t \mathbf{K}_t\left( \mathbf{z}_t - \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1}\right)$$

$$\mathbf{C}_t\hat{\mathbf{x}}_{t|t} = \mathbf{C}_t\hat{\mathbf{x}}_{t|t-1} + \mathbf{C}_t \mathbf{K}_t\left( \mathbf{z}_t - \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1}\right)$$

Combining the above two results we find that

$$\left(\begin{matrix} \mathbf{H}_t \\ \mathbf{C}_t\end{matrix} \right) \hat{\mathbf{x}}_{t|t} = \left(\begin{matrix} \mathbf{H}_t \\ \mathbf{C}_t\end{matrix} \right) \hat{\mathbf{x}}_{t|t-1} + \left(\begin{matrix} \mathbf{H}_t \\ \mathbf{C}_t\end{matrix} \right)  \mathbf{K}_t\left( \mathbf{z}_t - \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1}\right)$$

Since
$$\left(\begin{matrix} \mathbf{H}_t \\ \mathbf{C}_t\end{matrix} \right)$$
is an invertible matrix, it can be removed from both sides, and we obtain

$$\boxed{ \hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t\left( \mathbf{z}_t - \mathbf{H}_t\hat{\mathbf{x}}_{t|t-1}\right)}$$

Note that the covariance matrix can be obtained using the above equation as

$$\boxed{\Sigma_{t|t} = \left( \mathbf{I} - \mathbf{K}_t \mathbf{H}_t\right)\Sigma_{t|t-1} \left( \mathbf{I} - \mathbf{K}_t \mathbf{H}_t\right)^\top + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^\top}$$
