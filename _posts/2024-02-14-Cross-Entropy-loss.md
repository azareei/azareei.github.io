---
layout: post
title: Cross Entropy Loss Derivation
date: 2024-02-14 10:30:00
description: Cross Entropy Loss
---

[Cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) is defined as 

$$L = \frac{1}{N} \sum_{n=1}^N l_n, \qquad l_n = -  \frac{\exp(x_{n,c})}{\sum_i \exp(x_{n,i})}, \text{where } y_n=c $$ 

where $$x$$ is the input, $$y$$ is the target, $$C$$ is the number of classes, $$N$$ is the mini-batch size. So the network for each case of $$n$$, predicts values of $$[x_{n,1}, ..., x_{n,C}]^T$$, and we pass it to log-soft-max, and then depending on what class it belongs to $$y_{n,c}$$ use that value, and then average over all the cases of mini-batch $$n=1, ..., N$$.  Note that $$y_{n} $$ here represents the group that it belongs to, in terms of one-hot vector it can also be written as 

$$L = \frac{1}{N} \sum_{n=1}^N l_n, \qquad l_n = -  \sum_{c=1}^C \frac{\exp(x_{n,c})}{\sum_i \exp(x_{n,i})} y_{n,c} $$

where here we used the one-hot representation of $$y_n$$. 

Our goal here is to do a derivation, to show why is the cross-entropy loss is defined as above. 



The Kullback-Leibler (KL) divergence between the two probability distribution $$q(z)$$ and $$p(z)$$ is defined as 

$$ D_{KL}[q||p] = \int_{-\infty}^{\infty} q(z) \log \frac{q(z)}{p(z)} dz $$ 

Now consider that we observe an empirical data $$\{y_i\}_{i=1}^{N}$$ (which are the classes for each case of the data). We can consider the output distribution is a weighted sum of the point masses as 

$$ q(y) = \frac{1}{N} \sum_{i=1}^N \delta (y-y_{i})$$

where $$\delta(\cdot)$$ is the delta Dirac function. We want to minimize the KL divergence between the output of the neural network $$P(y|\theta)$$, and this empirical distribution, 

$$ \hat{\theta} = \arg\min_\theta \left[ \int_{-\infty}^{\infty} q(y) \log {q(y)} dy - \int_{-\infty}^{\infty} q(y) \log {p(y)} dy \right] $$ 

$$ \hat{\theta} = - \arg\min_\theta  \int_{-\infty}^{\infty} q(y) \log {P(y|\theta)} dy$$


Now, we replace for $$q(y)$$ to find 

$$ \hat{\theta} = - \arg\min_\theta \int_{-\infty}^{\infty} \left( \frac{1}{N} \sum_{n=1}^N \delta (y-y_{n}) \right)  \log {P(y|\theta)} dy $$

$$  \hat{\theta} = - \arg\min_\theta \frac{1}{N}  \sum_{n=1}^N \log {P(y_n|\theta)} $$ 

Note that the output of the network is  $$[x_{n,1}, ..., x_{n,C}]^T$$, that is transformed into probabilities using a soft-max function as 

$$ P(y_n|\theta) = \sum_{c=1}^C \frac{\exp(x_{n,c})}{\sum_i \exp(x_{n,i})} y_{n,c} $$ 

So as can be seen above the loss can be written as 


$$ L = \frac{1}{N} \sum_{n=1}^N l_n, \qquad l_n = -  \sum_{c=1}^C \frac{\exp(x_{n,c})}{\sum_i \exp(x_{n,i})} y_{n,c} $$ 

So that's it. Basically cross-entropy loss is the KL divergence between the point-mass distribution and the output probability prediction of the network (using a soft-max probability assignment). 