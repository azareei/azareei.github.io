---
layout: post
title:  Baye's rule in odds form
date: 2020-12-26 14:30:00
description: Baye's rule in odds form
---


Bayes rule discusses the probability of an event based on prior knowledge and new conditions that have occurred. It basically states the following equation

$$P(A\vert B) = \frac{P(A\cap B)}{P(B)} =  \frac{P(B\vert A)(A)}{P(B)} = \frac{P(B\vert A)(A)}{P(B\vert A)P(A) + P(B\vert \neg A) P(\neg A)} $$

where $$A$$ and $$B$$ are two events and $$P(B) \neq 0$$. The most commonly used form of Baye's rule is the last equality or 

$$P(A\vert B) = \frac{P(B\vert A)(A)}{P(B\vert A)P(A) + P(B\vert \neg A) P(\neg A)} $$

Let's use this equation for a cancer diagnosis test. We are interested to find the probability of having cancer given a positive test result. $$A,B$$ are then the events of *having cancer* and *testing positive*. The Baye's rule then becomes

$$P(C\vert +) = \frac{P(+\vert C)P(C)}{P(+\vert C)P(C) + P(+\vert \neg C) P(\neg C)}$$

where $$P(C)$$ is the probability of having cancer and $$P(\neg C)$$  is the probability of not having cancer. These two probabilities are priors or assumptions that we make about the initial rate of cancer in society and the test does not have to do anything with it. On the other hand, $$P(+\vert C)$$ is the true positive rate of the test, and $$P(+\vert \neg C)$$ is the false positive rate where they depend on the test and its performance.  All the terms (those that depend on the prior knowledge or the test performance) on the RHS are intertangled, and this makes it hard and sometimes counter-intuitive to work with Baye's rule to understand probabilities. One easy fix is to work with odds instead of probabilities. For example, if $$P(C) = 0.1$$, it means that among 10 people, the odds of having cancer to not having cancer is $$1:9$$. If we use the odds, Baye's rule becomes intuitive. We can manipulate Baye's rule to work with the odds 

$$\frac{P(C\vert +)}{1-P(C\vert +)} = \frac{\frac{P(+\vert C)P(C)}{P(+\vert C)P(C) + P(+\vert \neg C) P(\neg C)}}{1- \frac{P(+\vert C)P(C)}{P(+\vert C)P(C) + P(+\vert \neg C) P(\neg C)}} = \frac{P(+\vert C)P(C)}{P(+\vert \neg C)P(\neg C)}  = \left( \frac{P(C)}{P(\neg C)}\right) \left( \frac{P(+\vert C)}{P(+\vert \neg C)} \right)$$

In summary we found that

$$\frac{P(C\vert +)}{P(\neg C\vert +)} = \left( \frac{P(C)}{P(\neg C)}\right) \left( \frac{P(+\vert C)}{P(+\vert \neg C)} \right)$$

The left-hand side is the odds of having cancer given a positive test. On the right-hand side, the first part is the prior that doesn't depend on the test: it's the odds of having cancer. The second part is Baye's factor that depends on the test: the ratio between positive results given cancer (true positive rate) and positive results without cancer (false positive rate). To make the above equation simpler, we can define the odds of an event as 

$$O(A_1 : A_2) = \frac{P(A_1)}{P(A_2)}$$

As a result the Baye's rule in odds form becomes

$$O(C:\neg C\vert +) = O(C:\neg C) \cdot \frac{P(+\vert C)}{P(+\vert \neg C)}$$

Let's use the odds form in an example. Suppose $$1\%$$ of women have breast cancer. So the odds of having breast cancer among women is $$1:9$$. This is a prior assumption that we make. Now we would like to see how these odds change if one has a positive test result. Assume that we have developed a test that is $$90\%$$ accurate, $$P(+\vert C) = 0.90$$. This is usually called the *sensitivity* of the test, meaning that if one has breast cancer, $$90\%$$ of times this test correctly gives out a positive result (true positive rate of our test). Our test is also $$80\%$$ specific. This means that the true negative rate is $$P(-\vert \neg C) = 0.80$$. As a result, the false-positive rate becomes $$P(+\vert \neg C) = 0.2$$. Now, a woman tests positive, what are the odds of having cancer? 

$$O(C:\neg C\vert +) = \frac{P(C\vert +)}{P(\neg C\vert +)} = \left( \frac{0.1}{0.9} \right) \left( \frac{0.9}{0.2} \right) = \frac{1}{2} $$

So initially, the odds of having cancer was $$1:9$$. After the positive result, the odds are updated using Baye's rule and it becomes $$1:2$$. If the test becomes $$90\%$$ specific, the above equation immediately tells us that the odds updates to $$(1:9)\times(0.9/0.1) = 1:1$$. This makes Baye's rule more intuitive to work with.


