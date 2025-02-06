---
layout: post
title: Generative AI
date: 2025-01-07 14:00:00-0400
number headings: first-level 1, max 6, start-at 1, _.1.1
toc: 
beginning: true
---

A generative model is a *joint* probability distribution $p(x)$, for $x\in\mathcal{X}$ . It's a joint distribution because $x$ can be multidimensional where it consists of multiple variables  $(x_1, x_2, \ldots, x_n)$. 

We also have *conditional* generative model $p(x\vert c)$ in which the generative model would be conditioned on inputs or covariates $c\in C$.


## 1 Types of generative Models


*  **Probabilistic graphical models (PGM):** uses simple, often linear, mappings to map a set of interconnected latent variables $z_1, \ldots, z_L$ to observed variables $x_1, \ldots, x_D$. 
* **Deep Generative Models (DGM):** uses deep neural networks to learn a complex mapping from a single latent vector $z$ to the observed data $x$. Types of DGM are
	* **Variational Autoencoders (VAE)**
	* **AutoRegressive Models (ARM) models**
	* **Normalizing Flows**
	* **Diffusion Models** 
	* **Energy Based Models (EBM)**
	* **Generative Adversarial Networks (GAN)**

The following table summarizes the different generative models across different aspects (we'll discuss why in the next chapters):

| **Model**                                                     | **Density**                                                                                                                                         | **Sampling**                                                                                                                   | **Training**                                                                                                                | **Latents**                                                                                                                      | **Architecture**                                                                                                                   |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **PGM-D**<br>i.e., Probabilistic Graphical Model - Directed   | Exact, fast: The joint distribution $p(x)$ can be computed exactly and efficiently, leveraging the directed graph structure.                        | Fast: Sampling is efficient using ancestral sampling, which sequentially samples variables based on dependencies in the DAG.   | MLE: Trained using Maximum Likelihood Estimation, which directly optimizes the likelihood of observed data.                 | Optional: Latent variables (hidden variables) can be included but are not required for the model.                                | Sparse DAG: The model uses a sparse Directed Acyclic Graph, where edges capture directed dependencies.                             |
| **PGM-U**<br>i.e., Probabilistic Graphical Model - Undirected | Approximate, slow: The joint distribution $p(x)$ requires approximations due to the intractable partition function, making it computationally slow  | Slow: Sampling typically involves computationally expensive methods like Markov Chain Monte Carlo (MCMC).                      | MLE-A: Trained using approximate Maximum Likelihood Estimation, as exact computation of likelihood is infeasible.           | Optional: Latent variables can be included but are not mandatory for the model.                                                  | Sparse graph: The model uses a sparse undirected graph, where edges represent mutual dependencies.                                 |
| **VAE**<br>i.e., Variational Auto Encoder                     | LB, fast: Provides a **lower bound** on the likelihood (e.g., Evidence Lower Bound, or ELBO) and is computationally efficient for density modeling. | Fast: Efficient sampling is achieved via reparameterization, enabling smooth gradient-based optimization in latent space.      | MLE-LB: Trained by maximizing a lower bound on the likelihood, balancing reconstruction and latent regularization.          | $\mathbb{R}^L$: Latent representations (e.g., a compressed representation of the data) are a central feature of the model.       | Encoder-Decoder: Uses an encoder to map data to latent space and a decoder to reconstruct the data from the latent space.          |
| **ARM**<br>i.d., AutoRegressive Model                         | Exact, fast: The joint distribution $p(x)$ is computed exactly and efficiently using the sequential nature of the model.                            | Slow: Sampling is sequential, requiring one variable to be sampled at a time, which increases computation time.                | MLE: Trained using Maximum Likelihood Estimation, directly optimizing the likelihood of sequentially modeled variables.     | None: Does not use latent variables; it explicitly models the observed data.                                                     | Sequential: Processes data one variable at a time, reflecting the sequential dependency structure of the model.                    |
| **Flows**                                                     | Exact, slow/fast: Exact computation of $p(x)$, but speed depends on the specific normalizing flow architecture and its invertible transformations.  | Slow: Sampling involves applying invertible transformations, which can be computationally expensive depending on the model.    | MLE: Trained using Maximum Likelihood Estimation by directly optimizing the likelihood of the transformed data.             | $\mathbb{R}^D$: Latent representations are central, as flows map data to and from latent space using invertible transformations. | Invertible: Uses invertible transformations to map between data and latent space, ensuring exact density computation.              |
| **EBM**, <br>i.e., Energy Based Models                        | Approx, slow: Density estimation is approximate due to the need to compute a complex energy-based objective, which is computationally expensive.    | Slow: Sampling often relies on expensive iterative methods like Langevin Dynamics to generate samples.                         | MLE-A: Trained using approximate Maximum Likelihood Estimation due to challenges in normalizing the energy function.        | Optional: Latent variables can be included but are not essential for energy-based models.                                        | Discriminative: Models the energy function to differentiate between observed and unobserved data rather than direct probabilities. |
| **Diffusion**                                                 | LB: Provides a **lower bound** on likelihood during training by modeling a sequence of forward and reverse processes.                               | Slow: Sampling involves iterative denoising steps (e.g., reversing the diffusion process), which is computationally intensive. | MLE-LB: Trained by maximizing a lower bound on the likelihood, optimizing the reconstruction of data from corrupted inputs. | $\mathbb{R}^D$: Latent representations are central, as the diffusion process maps data into progressively noisier latent spaces. | Encoder-Decoder: Uses an encoder to add noise to data (diffusion) and a decoder to reverse the process (denoising).                |
| **GAN**, <br>i.e., Generative Adversarial Networks            | NA: Does not explicitly model the density $p(x)$; instead, it learns to generate data by adversarial training.                                      | Fast: Sampling is efficient, as the generator directly maps random noise to generated data in one forward pass.                | Min-max: Trained using adversarial training, where a generator and discriminator compete to improve data generation.        | $\mathbb{R}^L$: Latent representations (e.g., random noise vectors) are central to generating data.                              | Generator-Discriminator: Combines a generator (to create data) and a discriminator (to evaluate its realism).                      |


## 2 Goals of Generative AI

1. **Data Generation**: One of the primary goals of generative AI is **data generation**, where models create new data samples that resemble the original data they were trained on. This includes generating realistic images, text, audio, or other forms of data. Generative AI is also used for various tasks, such as:
	- **Creating synthetic data** for training discriminative models.
	- **Conditional generation** to control outputs based on specific inputs, enabling applications such as:
	    - Text-to-image (e.g., generating an image from a text description).
	    - Image-to-text (e.g., image captioning).
	    - Image-to-image (e.g., colorization, inpainting, uncropping, and restoration).
	    - Speech-to-text (e.g., automatic speech recognition or ASR).
	    - Sequence-to-sequence (e.g., machine translation or text continuation).

	The difference between conditional generative models and discriminative models lies in the outputs: generative models allow multiple valid outputs, whereas discriminative models assume a single correct output.
	
2.  **Density estimation**: Generative models are useful for calculating the probability of observed data, $p(x)$, which is known as density function. This has applications in: 
	- **Outlier detection**: Identifying data points with low probability under the estimated distribution, which may indicate anomalies or rare events. 
	- **Data compression**: Using the probability distribution to represent data more efficiently by assigning shorter codes to more likely outcomes.
	- **Generative classification**: Classifying data by estimating class-conditional densities $p(x\vert c)$ and combining them with prior probabilities to make decisions.
	- **Model comparison**: Evaluating and comparing different models by analyzing how well they represent the observed data distribution.
	  
	  Simple methods like kernel density estimation (KDE) are effective in low dimensions, using kernels (e.g., uniform or Gaussian) to estimate $p(x\vert \mathcal{D})$ based on observed data $\mathcal{D}$. However, KDE faces challenges in high dimensions due to the curse of dimensionality, requiring the use of parametric models  $p_\theta(x)$ for efficient and scalable density estimation.
	
3. **Missing Data Imputation**: missing values in a dataset are filled in using probabilistic methods. For example, a simple approach like mean value imputation replaces missing values with the average of observed values for each feature but ignores dependencies between variables. A more advanced method involves fitting a generative model to the observed data  $p(X_o)$ and sampling missing values conditioned on the observed data $p(X_m \vert X_o)$. This approach, called multiple imputation, can handle complex data types, such as filling in missing pixels in images (in-painting). Generative models provide a more robust solution by capturing variable dependencies and offering uncertainty estimates for the imputed values.
   
4. **Structure Discovery**: Generative models with latent variables can uncover hidden patterns or structures in data by inferring the latent causes $z$ that generate observed data $x$ using Bayes’ rule $p(z\vert x) \propto p(z)p(x\vert z)$. For example, in social network analysis, a generative model can represent each user’s behavior and interactions using latent variables $z$  that correspond to hidden community memberships or shared interests. By applying Bayes’ rule, the model can infer these latent communities $z$ from observed interaction patterns $x$, revealing the underlying structure of the network and grouping users with similar behaviors or preferences.
   
5.  **Latent Space Interpolation**: Generative models with latent variable representations enable **latent space interpolation**, where new data is generated by smoothly blending features between existing data points. For example, given two images, their latent encodings $z_1$ and$ z_2$ can serve as anchors in the latent space. By interpolating linearly between these encodings $z = \lambda z_1 + (1-\lambda)z_2$ and decoding the results, the model generates images combining characteristics of both inputs, such as transitioning between two digits or blending facial attributes. Additionally, **latent space arithmetic** allows modifying specific attributes. For example, in a model trained on celebrity faces, adding a learned offset vector for “sunglasses” to a latent encoding can generate a version of the face with sunglasses, while subtracting the vector can remove them. These techniques make latent spaces powerful for generating and manipulating data with meaningful variations.
   
6. **Generative Design**: Generative models are used to explore designs with specific properties. For example, in material science, a model trained on molecular data can generate new chemical structures optimized for properties like higher conductivity or stability by searching the latent space.

7. **Model-Based Reinforcement Learning**: Generative models simulate environments, such as robotics tasks, enabling agents to practice and plan in virtual settings, reducing reliance on expensive real-world data collection.

8. **Representation Learning**: Generative models learn compact latent representations $z$ that capture the underlying structure of data. For example, a model trained on medical images can extract features like the presence of tumors, which can then be used for tasks like diagnosis or prediction.

9. **Data Compression**: Generative models predict the probability of data patterns and assign shorter codes to frequent patterns, enabling efficient storage and transmission of data, as described by Shannon’s information theory.


## 3 Evaluating generative models

The generative AI models are evaluated on three aspects 
1. **Sample Quality**: Do the generated examples look realistic and belong to the same type of data as the training set?
2. **Sample Diversity**: Do the generated examples represent all the different variations present in the real data?
3. **Generalization**: Can the model create new examples that go beyond simply memorizing the training data?

No single metric captures all above, we use different metrics for each or combine parts. 

### 3.1 Likelihood-based evaluation
To evaluate how well a generative model $q$ matches the true data distribution $p$, **KL Divergence** is a commonly used metric:

$$
D_{KL}(p \parallel q) = \int p(x) \log \frac{p(x)}{q(x)} dx = \int p(x) \log p(x) \, dx - \int p(x) \log q(x) \, dx = - H(p) + H_{ce}(p, q)
$$
  

  The $D_{KL}$ measures the *distance* between the two distributions. Smaller value indicates that the model closely approximates the true data distribution. The first term on the RHS is the **entropy** of $p(x)$, denoted as $H(p)$. Since $H(p)$ depends only on the true distribution $p(x)$, it is a constant when evaluating the model $q(x)$. The second term is the **cross-entropy** between $p(x)$ and $q(x)$, denoted as $H_{ce}(p, q)$.  
  
  Minimizing $D_{KL}(p \parallel q)$ is equivalent to minimizing the cross-entropy $H_{ce}(p, q)$ (note that the entropy term,  $H(p)$, is constant and does not depend on $q$).
  
  Given an empirical observed dataset $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$, we can approximate $p(x)$ with the empirical data distribution,  $p(x) = \frac{1}{N} \sum_{i=1}^N \delta (x-x_i)$,  where $x_i$ is the $i$th observed data, and $\delta(.)$ is the Dirac's delta function. The cross-entropy then becomes:
  

$$H_{ce}(p, q) = -\frac{1}{N} \sum_{n=1}^{N} \log q(x_n)
$$

This is known as the **Negative Log Likelihood (NLL)**.

$$\text{NLL} = -\frac{1}{N} \sum_{n=1}^{N} \log q(x_n)
$$

Key Insights: 
* **NLL and Cross-Entropy**: The NLL represents the average penalty for the model $q(x)$ assigning probabilities to observed data points in a dataset.
  
* **Test Set Evaluation**: NLL is typically computed on a held-out test set to measure the model’s generalization ability.
  
* **Entropy** $H(p)$: Since entropy is a constant with respect to the model $q(x)$, it does not affect optimization when training the model.


#### 3.1.1 NLL and perplexity 

For models of discrete data, such as language models, **Negative Log Likelihood (NLL)** is a straightforward way to measure how well the model predicts the data. NLL evaluates the average “surprise” the model experiences when it encounters the actual outcomes in the dataset, based on the probabilities it assigns to them. The term “surprise” refers to how unexpected an event is, given the model’s prediction. If the model assigns a high probability to the correct outcome, the surprise (and NLL) is low. Conversely, if the model assigns a low probability, the surprise is high, reflecting the model’s uncertainty.
 * **Example:**	
	Suppose a language model predicts the next word in the sentence “The cat sat on the __” with the following probabilities:

		$$q(\text{“mat”}) = 0.6, \quad q(\text{“floor”}) = 0.3, \quad q(\text{“table”}) = 0.1$$
		
	If the correct word is “mat,” the NLL for this prediction is simply:
	
	$$\text{NLL} = -\log_2(q(\text{“mat”})) = -\log_2(0.6) \approx 0.737
	$$
	
	The NLL for a dataset averages these values across all predictions, measuring how well the model predicts the actual words.

Interpreting NLL directly can be unintuitive. To make it easier to understand, **perplexity** is used. Perplexity translates NLL into a measure that reflects how “confused” the model is—essentially, the average number of equally likely choices the model is effectively guessing from.
* **Why Does Perplexity Represent Choices?**: If a model has a perplexity of $P$, it behaves as if it is guessing from $P$ equally likely options. This comes from the relationship between NLL and uniform distributions: for $P$ equally likely outcomes, $q(x) = 1/P$, and the NLL is:
  
  $$\text{NLL} = -\log_2(1/P) = \log_2(P)$$
  
  Inverting this gives $$P = 2^{\text{NLL}}$$

Mathematically, perplexity is defined as:
  $$\text{Perplexity} = 2^H$$

where $H = \text{NLL}$. Lower perplexity indicates better performance, as the model is less “confused” and more confident in its predictions.


##### 3.1.1.1 More about KL divergence

The KL Divergence is defined as:

$$D_{KL}(p \parallel q) = \int p(x) \log \frac{p(x)}{q(x)} dx
$$
  
This measures the average difference in log probabilities between the true distribution $p(x)$ and the model distribution $q(x)$, weighted by $p(x)$. Two interpretations: 
1. **Information Loss**: $D_{KL}$ quantifies how much information is lost when $q(x)$(the model) is used to approximate $p(x)$ (the ground truth): In information theory, the information content (or “surprise”) of an event $x$ occurring under a probability distribution $p(x)$ is given by $-\log p(x)$. So $D_{KL}$ is the information difference when we use the model $q(x)$ instead of true model $p(x)$, i.e.,  $-\log q(x) - (-\log p(x))$, weighted and averaged by the by $p(x)$. 
2. **Encoding**: How inefficient is it to encode samples from $p(x)$ using a code optimized for $q(x)$: In information theory, the length of a code for an event $x$ is proportional to $-\log p(x)$, minimizing the average code length (Shannon’s Source Coding Theorem). If you use a code based on $q(x)$ instead of $p(x)$, the expected length of the code will increase. The extra cost per event is: $\log \frac{p(x)}{q(x)} = \log p(x) - \log q(x)$. Then, the expected extra cost is the KL divergence: $D_{KL}(p \parallel q) = \int p(x) \big[\log p(x) - \log q(x)\big] dx$. 

#### 3.1.2 Handling Continuous Data:

In image and audio data, we have the following challenge with likelihood:

* The **model** typically represents the data using a **continuous probability density function (PDF)** $p(x)$, where $x$ can take any real value.
* However, the **data** itself is discrete (e.g., pixel intensities are integers from 0 to 255).

Since a PDF can take values greater than 1, the average log-likelihood for discrete data can become arbitrarily large, making direct evaluation difficult. To address this, **uniform dequantization** is used. 

##### 3.1.2.1 Dequantization when handling Continuous Data   

Dequantization is a method used in probabilistic modeling to handle discrete data (e.g., pixel intensities $0–255$) with continuous probability density functions (PDFs), such as in image and audio models. Directly modeling discrete data with continuous PDFs can lead to degenerate solutions where arbitrarily high likelihoods are assigned to discrete points. To mitigate this, uniform random noise is added to discrete values, transforming them into continuous values. This process avoids undefined densities and provides a lower bound for the discrete log-likelihood. Following are steps to take 

* **Input**: Pixel values in $\{0, 1, ..., 255\}$.
- **Dequantization**:
    - Add uniform noise: $z = x + \mathcal{U}(0, 1)$.
    - Normalize: $z = z / 256$, resulting in $z \in [0, 1]$.
- **Flow Transformations**:
    - Apply transformations like sigmoid scaling: $z = \sigma^{-1}\left(\frac{z - 0.5\alpha}{1 - \alpha}\right)$, mapping data from $[0, 1]$ to $(-\infty, \infty)$.
    - Use flow layers (e.g., affine coupling layers) to map $z$ to a latent Gaussian space.
- **Training Objective**:
    - Maximize likelihood: 
		$\log p(x) \geq \mathbb{E}_{q(z\vert x)} \left[\log p(z) - \log q(z\vert x)\right]$
		
		Proof: 
		
		$p(x) = \int p(x\vert z)p(z)dz = \int q(z\vert x) \frac{p(x\vert z)p(z)}{q(z\vert x)}dz$
		
		$\log p(x) = \log \int q(z|x) \frac{p(x|z)p(z)}{q(z|x)} \, dz$
		
		Jensen's inequality states that for a convex function $f$  is  
		
		$f\left(\mathbb{E}[X]\right) \leq \mathbb{E}[f(X)]$
		
		Since the logarithm is a concave function, we can apply Jensen's inequality:
		
		$\log p(x) \geq \int q(z\vert x) \log \frac{p(x\vert z)p(z)}{q(z\vert x)} \, dz$
		
		Expand the term inside the logarithm:
		
		$\int q(z\vert x) \log \frac{p(x\vert z)p(z)}{q(z\vert x)} \, dz = \int q(z\vert x) \left[\log p(x\vert z) + \log p(z) - \log q(z\vert x)\right] \, dz$
		
		The first term, $\log p(x\vert z)$, integrates to zero because $p(x\vert z) = \delta(x - \text{round}(z))$, and $q(z|x)$ is only defined over valid $z$. As a result 
		
		$\log p(x) \geq \mathbb{E}_{q(z\vert x)} \left[\log p(z) - \log q(z\vert x)\right]$
		
		**The prior likelihood term**: $\log p(z)$ encourages the latent variable $z$ (the output of the flow transformations applied to the input $x$) to follow a predefined distribution, such as a Gaussian.
		**The dequantization likelihood term**: $-\log q(z\vert x)$ models the distribution $q(z\vert x)$, which is the distribution of the dequantized variable $z$ given the discrete input $x$.

#### 3.1.3 Likelihood can be hard to compute

In many generative models, we want to compute the **likelihood**  $p(x)$ , which tells us how well the model explains the data  $x$. Computing  $p(x)$  often requires evaluating a **normalization constant**

  $p(x) = \frac{\tilde{p}(x)}{Z}, \quad Z = \int \tilde{p}(x) dx$

where $\tilde{p}(x)$  is an unnormalized probability, and  $Z$  ensures the total probability integrates to $1$. Computing  $Z$  involves an integral over the entire data space, which can be very expensive, especially for high-dimensional data (e.g., images or text).

##### 3.1.3.1 Example

In a model with latent variables  $z$ , $ p(x)$  is computed as:  

$p(x) = \int p(x\vert z)p(z) dz$

This requires integrating over all possible  $z$ , which can be computationally infeasible for complex models. We basically need to find all $z$ that result in that generated data $x$. 


##### 3.1.3.2 Solution 1: **Variational Inference**

**Variational Inference (VI)** is a technique to approximate  $p(x)$  without explicitly calculating the normalization constant. It introduces a simpler distribution $q(z\vert x)$  to approximate the true posterior  $p(z\vert x)$. Instead of directly computing  $\log p(x)$ , we compute a **lower bound** (called the ELBO):

We start with:
$p(x) = \int p(x, z) \, dz =  \int q(z|x) \frac{p(x, z)}{q(z\vert x)} \, dz$

Applying Jenson's inequality;

$\log p(x) = \log \int q(z\vert x) \frac{p(x, z)}{q(z\vert x)} \, dz \geq \int q(z\vert x) \log \frac{p(x, z)}{q(z\vert x)} \, dz$

Noting that: 

$\log \frac{p(x, z)}{q(z\vert x)} = \log p(x\vert z) + \log p(z) - \log q(z\vert x)$

We find 

$\log p(x) \geq \int q(z\vert x) \log p(x\vert z) \, dz + \int q(z\vert x) \log p(z) \, dz - \int q(z\vert x) \log q(z\vert x) \, dz$

Apply the KL Divergence

$\text{KL}(q(z\vert x) \| p(z)) = \int q(z\vert x) \log \frac{q(z\vert x)}{p(z)} \, dz$

Final ELBO

$\log p(x) \geq \mathbb{E}_{q(z\vert x)}[\log p(x\vert z)] - \text{KL}(q(z\vert x) \| p(z))$

This avoids the expensive integral over  $z$  and allows us to optimize the model using the lower bound. In the ELBO expression above
* $\mathbb{E}_{q(z\vert x)}[\log p(x\vert z)]$ : Encourages  $q(z\vert x)$  to explain the observed data well.
* $\text{KL}(q(z\vert x) \| p(z))$ : Ensures $q(z\vert x)$  stays close to the prior  $p(z)$ , avoiding overfitting.

For example, in Variational Autoencoders (VAEs), we:
1. Approximate the true posterior $p(z\vert x)$ with a simpler  $q(z\vert x)$ (like a Gaussian)
2. Optimize the ELBO to train the model without computing the exact  $p(x)$

##### 3.1.3.3 Solution 2: **Annealed Importance Sampling (AIS)**
In this method, we estimate the log likelihood using Monte Carlo sampling. 

#### 3.1.4 Likelihood and sample quality
A model can achieve high likelihood but produce perceptually poor samples and vice versa. Likelihood alone is not a reliable indicator of the perceptual quality of generated samples.

##### 3.1.4.1 Example of High Likelihood but Poor Samples
Consider the following
* Model $q_0$: A good density model that performs well in terms of average log-likelihood.
* Model $q_1$: A bad model that generates white noise.
Now imagine a mixture model $q_2$
• Mixture Model $q_2$: Combines these models:
$$q_2(x) = 0.01q_0(x) + 0.99q_1(x)$$
This means 99% of the samples will be poor (from $q_1$). Since $q_1$ is coming from uniform distribution (white noise), and for high-dimensional data (like images with many pixels) it would be a very small number, so negligible in comparison with $q_0$, so we will have 

$$  \log q_2(x) = \log[0.01q_0(x) + 0.99q_1(x)] \geq \log[0.01q_0(x)] = \log q_0(x) - 2$$
which would be a large log likelihood for $q_2$, however, the sample quality is pretty bad. 

##### 3.1.4.2 Example of Low Likelihood but Great Sample quality 
Imagine a Gaussian Mixture Model defined as 

$$  q(x) = \frac{1}{N} \sum_{n=1}^{N} N(x \vert x_n, \epsilon^2 I)$$
The model consists of a mixture of **N Gaussians**. Each **Gaussian**  $N(x \vert x_n, \epsilon^2 I)$  is centered on a training image  $x_n$ . Note that $\epsilon^2 I$  represents small Gaussian noise added around each training image. If  $\epsilon$  **is very small**, each Gaussian is tightly concentrated around the training images. This means that when we **sample from this model**, we get images that are **almost identical to training images**. **Perceptually**, the generated samples look great because they resemble real training images. Likelihood measures **how well the model generalizes to new data**. In this case, since the model is just a bunch of Gaussians centered on training images, its density function will **assign high probability to training images.** and **assign very low probability to test images,** which means **poor likelihood on the test set**.

### 3.2 Perceptual Metrics
Evaluating generative models (like GANs and VAEs) is challenging because traditional metrics like likelihood do not always reflect the perceptual quality of generated images. Instead of directly comparing raw pixel values, researchers use **perceptual distance metrics**, which compare feature representations of real and generated images. These features are extracted using neural networks, often from **pretrained classifiers** like the Inception model. 

1. **Inception Score (IS)**: This score measures how well a generative model produces diverse and recognizable images. It uses a classifier (like the Inception network) to predict class labels for generated images.  

$$IS = \exp (E_{p_\theta(x)} D_{KL} (p_{\text{disc}}(Y \vert x) \parallel p_\theta(Y)))
$$
	where:
	* $p_\theta(x)$  is the probability distribution of images generated by the model.
	* $p_{\text{disc}}(Y \vert x)$  is the probability distribution over class labels **given an image**  x . This is obtained from a **pretrained classifier** (like Inception).
	* $p_\theta(Y)$  is the **marginal class distribution** of generated images:
	
  $$p_\theta(y) = \int p_{\text{disc}}(y \vert x) p_\theta(x) dx$$
  

	What is the Inception Score (IS) trying to do? 
	
	The Inception Score is trying to answer two questions about the images generated by a model:
	1. **Are the generated images clear and recognizable?** (Do they belong to a specific class with high confidence?)
	2. **Are the generated images diverse?** (Do they cover a wide range of different classes?)
	   
	To measure this, it **compares two distributions**:
	* $p_{\text{disc}}(Y \vert x)$  -> This is the **classifier’s prediction** for a generated image  $x$ . It tells us how confident the classifier is that the image belongs to a certain class (e.g., "cat" or "dog").
	* $p_\theta(Y)$  -> This is the **overall distribution of generated class labels** across many images. It tells us whether the model is generating a balanced mix of different classes.
	  
	  
	The score is calculated using **KL divergence**, which measures how different these two distributions are. This happens in two steps
		1. **Check if each image belongs to a clear class.** --> If an image is confidently recognized as a **specific** class (e.g., “cat” with 99% probability), it is a **high-quality sample**. If an image is blurry or unrecognizable, the classifier will be **uncertain** which is **bad**.
		2. **Check if the model generates a variety of different classes.**: If the model generates **only cats**, it is **not diverse**, which is **bad**. If the model generates a **mix of different animals**, it is **diverse**, which is **good**.
		3. Combining the above two: **KL divergence compares how different the per-image class prediction**  $p_{\text{disc}}(Y \vert x)$  **is from the overall class distribution**  $p_\theta(Y)$ **.** If the two distributions are very different, it means the images are **recognizable and diverse**, which gives a **high score**. If the distributions are similar, it means the images are **blurry or all from the same class**, which gives a **low score**.
		   
	
	This can also be seen with a bit of derivation. The KL divergence in the inception score can be written as 
	$$D_{KL} (p_{\text{disc}}(Y \vert x) \parallel p_\theta(Y)) = \sum_y p_{\text{disc}}(y \vert x) \log \frac{p_{\text{disc}}(y \vert x)}{p_\theta(y)}$$
	
	So the expected value is 
		$$E_{p_\theta(x)} D_{KL} (p_{\text{disc}}(Y \vert x) \parallel p_\theta(Y)) = \int p_\theta(x) \sum_y p_{\text{disc}}(y \vert x) \log \frac{p_{\text{disc}}(y \vert x)}{p_\theta(y)} dx$$
		
	Rearranging, we obtain 
	$$E_{p_\theta(x)} D_{KL} (p_{\text{disc}}(Y \vert x) \parallel p_\theta(Y)) = \sum_y \int p_\theta(x) p_{\text{disc}}(y \vert x) \log \frac{p_{\text{disc}}(y \vert x)}{p_\theta(y)} dx$$
	
	Note that $p_\theta(y) = \int p_{\text{disc}}(y \vert x) p_\theta(x) dx$. So the above would mean
		$$E_{p_\theta(x)} D_{KL} (p_{\text{disc}}(Y \vert x) \parallel p_\theta(Y)) = H(p_\theta(Y)) - E_{p_\theta(x)} [H(p_{\text{disc}}(Y \vert x))]$$
		
	So in here 
	* $H(p_\theta(Y))$ is the **entropy of the marginal class distribution**: **High**  $H(p_\theta(Y))$ **(high marginal entropy) -> Ensures diversity**
	*  $H(p_{\text{disc}}(Y \vert x))$  is the **conditional entropy** of class predictions per image: **Low**  $H(p_{\text{disc}}(Y \vert x))$  **(low per-image entropy) -> Ensures realism**

	This represents the overall class distribution of the generated images.
	A high score means the model generates **varied** images across different classes (**high entropy** of predicted labels). Each individual image should be **easily classifiable** (low entropy per image). One big **limitation** is that it does not measure overfitting—if a model memorizes one perfect image per class, it can still score well.

2. **Fréchet Inception Distance (FID)**: Instead of class labels, it compares statistical properties of deep features (mean and covariance) between real and generated images.

$$FID = \|\mu_m - \mu_d\|^2_2 + \text{tr} (\Sigma_d + \Sigma_m - 2(\Sigma_d \Sigma_m)^{1/2})$$
  
	where 
	* $\mu_m, \Sigma_m$  = Mean and covariance of generated images
	* $\mu_d, \Sigma_d$  = Mean and covariance of real images
	Lower FID is better because it means generated images closely resemble real images in feature space. One **limitation** of this score is that it is sensitive to the number of samples used—results can vary due to statistical bias.

3. **Kernel Inception Distance (KID)**: This score improves on FID by using **Maximum Mean Discrepancy (MMD)** to measure the similarity between distributions. This reduces bias issues in FID by comparing feature distributions more robustly. MMD measures the distance between two distributions **without assuming they are Gaussian**. Instead of just comparing the **mean and covariance** (like FID), MMD measures how well two sets of samples **match** using a **kernel function**.   

$$\text{MMD}^2(X, Y) = E[k(x, x{\prime})] + E[k(y, y{\prime})] - 2E[k(x, y)]$$


	where:
	* $X$  = set of features from real images.
	* $Y$  = set of features from generated images.
	* $k(x, y)$  = a kernel function that measures similarity (usually a **polynomial or Gaussian kernel**).
	* $E[k(x, x{\prime})]$  = similarity within real images.
	* $E[k(y, y{\prime})]$ = similarity within generated images.
	* $E[k(x, y)]$  = similarity between real and generated image

### 3.3 Precision and recall metrics
The **Fréchet Inception Distance (FID)** measures the **distance between the real and generated data distributions** but does **not tell us why a model is failing**. A **bad (high) FID** could mean:
1. The model **produces low-quality samples** (they don’t look realistic).
2. The model **places too much probability mass around the data distribution**, meaning it only captures a limited part of the real data.
3. The model **only generates a subset of the real data** (a problem called **mode collapse** in GANs).
Precision (sample quality), and recall (sample diversity) are introduced to resolve this differentiation issue

* ***Precision** -> Are generated samples **high quality** (similar to real data)?
* * **Recall** -> Is the **diversity** of generated samples good (does it cover the full real data distribution)?

**Precision and recall** work by using a **pretrained classifier** (like Inception) to extract **features** of both real and generated images. Then, **nearest neighbor distances** are used to compare them. Let’s define the notation:
* $\Phi_{\text{model}}$  = Set of feature vectors from generated images.
* $\Phi_{\text{data}}$  = Set of feature vectors from real images.
*  $\text{NN}_k(\phi{\prime}, \Phi)$  = The  $k$-th nearest neighbor of  $\phi{\prime}$  in  $\Phi$  (used to measure how close a sample is to its nearest neighbors).
To determine whether a generated image is **close enough** to real data, we define the function:

$$f_k(\phi, \Phi) =

\begin{cases}

1, & \text{if } \exists \phi{\prime} \in \Phi \text{ such that } \|\phi - \phi{\prime}\|_2^2 \leq \|\phi{\prime} - \text{NN}_k(\phi{\prime}, \Phi)\|_2^2 \\

0, & \text{otherwise}

\end{cases}
$$

 This function
 * checks whether a generated sample $ \phi$  is as **close to real data as real data is to itself**
 * if $\phi$  is **close enough** to any real data point  $\phi{\prime}$ , it is counted as **a valid sample**.

**Precision (Sample Quality)**: Precision tells us **how many generated samples** are close to real data.

$$\text{precision}(\Phi_{\text{model}}, \Phi_{\text{data}}) = \frac{1}{|\Phi_{\text{model}}|} \sum_{\phi \in \Phi_{\text{model}}} f_k(\phi, \Phi_{\text{data}})$$

  * ***High precision** means most generated images are **high-quality** and resemble real data.
  * **Low precision** means many generated images are **not realistic**.

**Recall (Sample Diversity)**: Recall tells us **whether real data points are well represented** by the model. 

$$\text{recall}(\Phi_{\text{model}}, \Phi_{\text{data}}) = \frac{1}{|\Phi_{\text{data}}|} \sum_{\phi \in \Phi_{\text{data}}} f_k(\phi, \Phi_{\text{model}})$$

  * ***High recall** means the model generates **a diverse set of samples that cover all real data**. 
  * **Low recall** means the model **misses** important variations in the real dataset (**mode collapse**).
Now with precision and recall we can separate between the issues that the generative model has. For example, in **GANs**, **mode collapse** happens when the model only generates a **few types of images**.
* ***High precision but low recall** = The GAN generates **very realistic but repetitive** images (e.g., only faces of young white males).
* **Low precision but high recall** = The GAN generates **a wide variety of images, but many are blurry**.
### 3.4 Statistical Tests
A **two-sample test** is a **statistical method** used to determine whether two datasets (sets of samples) come from the **same underlying distribution**.
* ***Null Hypothesis (** $H_0$ **)**: The two sets of samples come from the **same** distribution.
* **Alternative Hypothesis (** $H_A$ **)**: The two sets of samples come from **different** distributions.
For example, in **Generative Adversarial Networks (GANs)** and other generative models, we want to check:
1. **Are the generated samples statistically similar to real data?**
2. **How different is the model’s output from real data?**

To do this, we apply two-sample tests using:
1. **Classifier-based statistics** -> Train a classifier to distinguish real vs. generated images.
2. **Maximum Mean Discrepancy (MMD)** -> A kernel-based test to compare distributions.

Since **deep learning works with high-dimensional data** (e.g., images with millions of pixels), two-sample tests often use **learned feature representations** (from pretrained networks like Inception) instead of comparing raw pixel values.

Statistical tests let users control **Type 1 error** ( $\alpha$ ), which is **the probability of wrongly rejecting the null hypothesis** (i.e., concluding that the samples are different when they are actually from the same distribution).

