---
layout: post
title: Generative AI
date: 2025-01-07 14:00:00-0400
---

A generative model is a *joint* probability distribution $p(x)$, for $x\in\mathcal{X}$ . It's a joint distribution because $x$ can be multidimensional where it consists of multiple variables  $(x_1, x_2, \ldots, x_n)$. 

We also have *conditional* generative model $p(x\vert c)$ in which the generative model would be conditioned on inputs or covariates $c\in C$.


## Types of generative Models


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


## Goals of Generative AI

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


## Evaluating generative models

The generative AI models are evaluated on three aspects 
1. **Sample Quality**: Do the generated examples look realistic and belong to the same type of data as the training set?
2. **Sample Diversity**: Do the generated examples represent all the different variations present in the real data?
3. **Generalization**: Can the model create new examples that go beyond simply memorizing the training data?