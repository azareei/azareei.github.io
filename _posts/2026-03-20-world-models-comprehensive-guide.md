---
layout: post
title: World Models - Learning to Simulate Reality
date: 2026-03-20 10:00:00-0400
description: A comprehensive exploration of world models from classical control theory to modern foundation models that learn to dream and predict the future
tags: machine-learning deep-learning reinforcement-learning computer-vision world-models
---

## Introduction

Imagine an agent that can predict what happens next when it takes an action—not just in simple games, but in the complex, messy real world with all its physics, dynamics, and uncertainties. This is the promise of **world models**: systems that learn internal representations of how the world works, enabling them to simulate futures, plan actions, and ultimately behave intelligently without exhaustive trial-and-error in reality.

While large language models (LLMs) have dominated AI headlines by predicting the next *word*, world models aim to predict the next *world state*. They're learning to understand causality, physics, and temporal dynamics—capabilities crucial for embodied AI systems like robots and autonomous vehicles that must interact with the physical world.

In 2026, world models are experiencing a renaissance. Major AI labs are releasing foundation world models that can generate interactive, physically consistent simulations. Companies like Waymo are using world models to train self-driving cars on scenarios that don't exist yet. And researchers are discovering that scaled video generation models exhibit emergent properties as "general purpose simulators of the physical world."

This post provides a comprehensive journey through world models: from their roots in classical control theory to cutting-edge transformer-based architectures that can dream up entire virtual worlds. We'll explore the mathematics, architectures, applications, and open challenges that define this exciting frontier of AI research.

## What Are World Models and Why Do They Matter?

### The Core Idea

A **world model** is a learned representation of environment dynamics—a function that predicts how the state of the world evolves over time, potentially conditioned on actions taken by an agent. Formally, given a current state $s_t$ and action $a_t$, a world model predicts the next state $s_{t+1}$ and potentially the reward $r_t$:

$$
s_{t+1}, r_t = f_{\text{world}}(s_t, a_t)
$$

But world models are more than just predictors. They're **simulators** that enable:

- **Planning**: Simulate multiple possible futures to find the best action sequence
- **Sample efficiency**: Learn from imagined experiences rather than costly real-world interactions
- **Counterfactual reasoning**: Explore "what if" scenarios without consequences
- **Transfer learning**: Build reusable knowledge about physics and dynamics

### Why World Models Matter Now

Several converging trends have made world models particularly relevant in 2026:

1. **Data efficiency imperative**: In robotics and autonomous driving, real-world data collection is expensive and dangerous. World models enable training in simulation.

2. **Scaling successes**: Just as scaling transformed language models, researchers are discovering that scaling world models on diverse video data produces emergent understanding of physics and causality.

3. **Embodied AI push**: As AI moves from text to physical interaction (robots, AVs, drones), understanding dynamics becomes essential.

4. **Generative model advances**: Breakthroughs in diffusion models and transformers have dramatically improved our ability to generate high-fidelity video predictions.

According to multiple AI research outlooks for 2026, world models are positioned as "the next big leap beyond LLMs" with industry interest multiplying as companies realize their potential for everything from robotics to video games.

## Historical Context: From Control Theory to Deep Learning

### Classical Foundations (1960s-1990s)

The idea of modeling system dynamics has deep roots in control theory and optimal control. The foundation was laid by several key developments:

**The Kalman Filter (1960)**: Hungarian mathematician Rudolf Emil Kalman revolutionized state estimation by introducing the Kalman filter, which maintains a probabilistic belief about a system's state by combining predictions from a dynamics model with noisy sensor measurements. The Kalman filter essentially maintains a "world model"—a set of states that evolve in time according to physical laws—while accounting for uncertainty and measurement noise.

The algorithm operates in two phases:
1. **Prediction**: Use the dynamics model to predict the next state: $\hat{x}_{t|t-1} = F_t \hat{x}_{t-1|t-1} + B_t u_t$
2. **Update**: Correct the prediction using measurements: $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - H_t\hat{x}_{t|t-1})$

where $K_t$ is the Kalman gain that optimally weights the prediction vs. measurement based on their respective uncertainties.

The Kalman filter and its nonlinear extensions (Extended Kalman Filter, Unscented Kalman Filter) became fundamental tools in aerospace, navigation, and robotics. NASA used nonlinear Kalman filters in its space program as early as the 1960s, with Stanley Schmidt driving their adoption.

**Model-Based Control**: Together with the Linear-Quadratic Regulator (LQR), the Kalman filter solved the Linear-Quadratic-Gaussian (LQG) control problem—arguably one of the most fundamental problems in control theory. This established the paradigm of using models for optimal decision-making under uncertainty.

### The Rise of Machine Learning (1990s-2000s)

As machine learning matured, researchers began replacing hand-engineered dynamics models with learned ones:

**System Identification**: Neural networks were applied to learn dynamics models from data, particularly for nonlinear systems where analytic models were intractable.

**Dyna Architecture (1990)**: Richard Sutton introduced the Dyna architecture, which unified model-free and model-based reinforcement learning. Dyna alternates between:
1. Collecting real experience and updating the model
2. Generating synthetic "imagined" trajectories using the model
3. Updating the policy using both real and imagined data

This established the key insight that models could amplify learning by generating synthetic experience for policy improvement.

**Forward Models in Neuroscience**: Research in computational neuroscience suggested that biological brains maintain internal models for motor control and prediction, inspiring similar approaches in AI.

### The Deep Learning Revolution (2010s-Present)

The combination of deep learning, large-scale video datasets, and computational power enabled a new generation of world models:

**Visual Prediction Models (2015-2018)**: ConvLSTM and PredRNN architectures showed that recurrent networks could predict future video frames by learning spatiotemporal dynamics. These models combined convolutional networks for spatial feature extraction with LSTMs for temporal modeling.

**World Models Paper (2018)**: David Ha and Jürgen Schmidhuber's influential "World Models" paper demonstrated that agents could learn to play games by training entirely inside learned world models—even on hallucinated "dream" data.

**Transformer Era (2020s)**: The success of transformers in language and vision inspired their application to world modeling, enabling long-range temporal dependencies and better scalability.

**Foundation World Models (2024-2026)**: Models like Google's Genie, OpenAI's Sora, and NVIDIA's Cosmos represent a paradigm shift: large-scale world models trained on diverse internet video that can generate interactive, controllable simulations.

## Fundamentals: Core Concepts and Mathematics

### State Space and Representations

At the heart of world modeling is the concept of **state**—a representation that captures all relevant information about the environment at a given time.

**Observable vs. Latent States**: In many real-world scenarios, we don't have direct access to the true state. Instead, we receive high-dimensional observations (like images) from which we must infer the underlying state:

$$
o_t = g(s_t) + \epsilon_t
$$

where $o_t$ is the observation, $g$ is the observation function, and $\epsilon_t$ represents noise.

**State Representation Learning**: A key challenge in modern world modeling is learning compact state representations from raw sensory data. This typically involves:
- **Encoding**: Map high-dimensional observations to low-dimensional latent states: $z_t = \text{Encoder}(o_t)$
- **Decoding**: Reconstruct observations from latent states: $\hat{o}_t = \text{Decoder}(z_t)$
- **Dynamics**: Model transitions in latent space: $z_{t+1} = f(z_t, a_t)$

Working in learned latent space is crucial for efficiency—predicting pixel-level futures directly is computationally expensive and focuses on irrelevant details.

### Deterministic vs. Stochastic Dynamics

World models must capture the inherent stochasticity in real-world dynamics:

**Deterministic Models**: Predict a single next state:
$$
s_{t+1} = f(s_t, a_t)
$$

These are simple but can't capture multi-modal outcomes (e.g., a ball rolling left or right after hitting a corner).

**Stochastic Models**: Predict a distribution over possible next states:
$$
p(s_{t+1} | s_t, a_t)
$$

Common approaches include:
- **Gaussian distributions**: $s_{t+1} \sim \mathcal{N}(\mu(s_t, a_t), \Sigma(s_t, a_t))$
- **Mixture models**: Model multi-modal distributions as weighted combinations
- **Implicit models**: Use generative models (VAEs, diffusion models) to sample from learned distributions

The **Recurrent State Space Model (RSSM)** used in the Dreamer series cleverly decomposes state into deterministic and stochastic components:
- **Deterministic state** $h_t$: Summarizes history via a recurrent network
- **Stochastic state** $z_t$: Captures unpredictable variations

This hybrid approach balances the ability to remember information across time steps (deterministic) with the flexibility to capture uncertainty (stochastic).

### Model-Based vs. Model-Free Learning

The use of world models sits within the broader context of model-based reinforcement learning (MBRL):

**Model-Free RL**: Learn a policy or value function directly from experience, without an explicit world model. Examples: DQN, PPO, SAC.
- **Pros**: Simple, direct optimization; works well with sufficient data
- **Cons**: Sample inefficient; poor generalization

**Model-Based RL**: Learn a world model, then use it for planning or to generate synthetic data for policy learning. Examples: MuZero, DreamerV3, Dyna.
- **Pros**: Sample efficient; enables planning; better generalization
- **Cons**: Model errors can compound; more complex to implement

The central challenge in MBRL is the **model error problem**: inaccuracies in the learned model can mislead policy learning, especially over long horizons. This has driven much recent research on robust world modeling.

### Partial Observability and Belief States

Real-world environments are often **partially observable**—the agent can't see everything (e.g., objects behind walls, internal states of other agents). This is formalized as a Partially Observable Markov Decision Process (POMDP).

In POMDPs, the agent must maintain a **belief state**—a distribution over possible true states given its observation history:
$$
b_t(s) = P(s_t = s | o_1, a_1, \ldots, o_t)
$$

Recurrent networks (LSTMs, GRUs) and attention mechanisms naturally maintain implicit belief states by aggregating information over time, making them well-suited for world modeling in partially observable settings.

## The Original World Models: Ha & Schmidhuber (2018)

The 2018 "World Models" paper by David Ha and Jürgen Schmidhuber crystallized many ideas and demonstrated their power in a compelling way. Let's examine this influential work in detail.

### Motivation and Setup

Ha and Schmidhuber tackled the CarRacing-v0 environment from OpenAI Gym—a task requiring visual perception and motor control. Their key insight: instead of learning a monolithic end-to-end policy, decompose the problem into perception, dynamics modeling, and control.

### Architecture: Vision, Memory, and Controller

The World Models architecture consists of three components:

**1. Vision (V): Variational Autoencoder**

The VAE compresses high-dimensional visual observations (96×96 RGB images) into a compact latent vector $z_t$ of dimension 32:

$$
\begin{align}
z_t &\sim q_\phi(z | o_t) = \mathcal{N}(\mu_\phi(o_t), \Sigma_\phi(o_t)) \\
\hat{o}_t &= p_\theta(o | z_t)
\end{align}
$$

The VAE is trained with the standard ELBO objective:
$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi}[\log p_\theta(o | z)] - \beta \cdot D_{\text{KL}}(q_\phi(z|o) \| p(z))
$$

Using a VAE (rather than a standard autoencoder) provides:
- A smooth latent space amenable to dynamics modeling
- Uncertainty estimates through the learned distribution
- Regularization against overfitting

**2. Memory (M): MDN-RNN**

The memory component predicts temporal dynamics in latent space using a Mixture Density Network combined with an LSTM:

$$
p(z_{t+1} | z_t, a_t, h_t) = \sum_{k=1}^K \pi_k(z_t, a_t, h_t) \cdot \mathcal{N}(\mu_k(z_t, a_t, h_t), \Sigma_k(z_t, a_t, h_t))
$$

where $h_t$ is the LSTM hidden state, and the mixture of Gaussians captures multi-modal dynamics.

The MDN-RNN is trained to predict the next latent code and reward:
$$
\mathcal{L}_{\text{MDN-RNN}} = -\mathbb{E}[\log p(z_{t+1}, r_t | z_t, a_t, h_t)]
$$

The mixture density network is crucial because a single Gaussian would average over multiple possible futures, producing blurry predictions.

**3. Controller (C): Linear Policy**

Remarkably, the controller that selects actions is a simple linear function of the concatenated VAE latent code and RNN hidden state:

$$
a_t = W_c [z_t, h_t] + b_c
$$

This tiny controller (only 867 parameters!) is trained using evolutionary algorithms (CMA-ES) to maximize cumulative reward.

### Training Procedure

Training proceeds in three independent stages:

1. **Train VAE**: Collect random rollouts, train VAE to reconstruct observations
2. **Train MDN-RNN**: Using VAE-encoded latent codes, train RNN to predict next latent and reward
3. **Train Controller**: Use evolutionary algorithms to optimize controller parameters, evaluating fitness in the real environment

### Learning Inside Dreams

The most striking result: agents can be trained entirely inside hallucinated "dreams" generated by the world model:

1. The MDN-RNN generates synthetic rollouts: $z_{t+1}, r_t \sim p(z_{t+1}, r_t | z_t, a_t, h_t)$
2. The controller is optimized using only these synthetic experiences
3. The resulting policy transfers to the real environment!

This demonstrates that the world model has captured sufficient environment structure for meaningful policy learning, despite never exactly matching reality.

### Impact and Limitations

The World Models paper had enormous influence, demonstrating:
- The power of learning in latent space
- That simple controllers can solve complex tasks when given good representations
- The feasibility of training in simulated environments

However, limitations included:
- Three-stage training is cumbersome and suboptimal
- No joint training or gradient flow between components
- Limited to relatively simple environments
- Model errors accumulate over long horizons

These limitations motivated subsequent work, particularly the Dreamer series.

## Modern Architectures: From Transformers to Diffusion Models

The field has evolved rapidly since 2018, with several architectural paradigms emerging for world modeling.

### Transformer-Based World Models

Transformers' ability to model long-range dependencies and scale effectively has made them a natural choice for world modeling.

**Autoregressive Transformers**: Models like GPT predict the next token in a sequence. Applied to world modeling, this becomes predicting the next frame or latent state token:

$$
p(s_{1:T} | a_{1:T}) = \prod_{t=1}^T p(s_t | s_{<t}, a_{\leq t})
$$

**Key advantages**:
- Flexible context length via attention
- Can incorporate multimodal inputs (images, actions, text)
- Leverage pre-trained vision-language models

**Examples**:
- **DrivingGPT**: Uses autoregressive transformers for multimodal driving world models, learning joint world modeling and planning through standard next-token prediction
- **Genie**: Google's foundation world model uses transformers to generate interactive environments from prompts

**Tokenization Challenge**: To apply transformers effectively, continuous states/images must be tokenized. Common approaches:
- **VQ-VAE**: Discretize latent space into learned codebook
- **Patch embeddings**: Treat image patches as tokens
- **Action tokens**: Interleave image and action tokens in the sequence

### Diffusion Models for World Modeling

Diffusion models have achieved state-of-the-art results in image and video generation, making them natural candidates for world modeling.

**Diffusion Process**: Instead of directly predicting the next state, diffusion models learn to iteratively denoise from pure noise to the target state:

$$
\begin{align}
\text{Forward (noising)}: && q(x_t | x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I) \\
\text{Reverse (denoising)}: && p_\theta(x_{t-1} | x_t) &= \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\end{align}
$$

**Conditional Generation**: For world modeling, we condition the denoising process on past states and actions:
$$
\epsilon_\theta(x_t, t, s_{<t}, a_{\leq t})
$$

**Diffusion World Models**: Research has shown that diffusion models can effectively model environment dynamics, particularly for:
- High-fidelity visual prediction
- Capturing multi-modal futures
- Handling complex, contact-rich dynamics

**Classifier-Free Guidance**: To incorporate action conditioning, diffusion world models use classifier-free guidance:
$$
\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, a) - \epsilon_\theta(x_t, \emptyset))
$$

where $w$ controls the strength of action conditioning.

**Diffusion Transformers (DiT)**: Recent work combines the strengths of both architectures:
- Use transformer backbone for the denoising network
- Apply to video tokens for scalable world modeling
- **UrbanDiT**: A foundation model for urban spatiotemporal prediction that successfully scales up diffusion transformers for open-world learning

**Trade-offs**: Diffusion models offer high sample quality but are slower at inference due to iterative denoising. This has driven research into faster sampling methods for real-time applications.

### Video Prediction Models

A parallel line of research has focused specifically on predicting future video frames—essentially world models in pixel space.

**ConvLSTM (2015)**: Pioneered spatiotemporal modeling by extending LSTMs with convolutional operations:
$$
\begin{align}
i_t &= \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + b_f) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tanh(W_{xc} * X_t + W_{hc} * H_{t-1} + b_c) \\
o_t &= \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + b_o) \\
H_t &= o_t \odot \tanh(C_t)
\end{align}
$$

where $*$ denotes convolution and $\odot$ element-wise multiplication.

**PredRNN Series (2017-2019)**: Introduced spatiotemporal memory cells that propagate information both horizontally (across time) and vertically (across layers), addressing the "deep-in-time" dilemma:

- **Spatiotemporal LSTM**: Dual memory structure for better spatiotemporal modeling
- **PredRNN++**: Added gradient highway units and Causal LSTM to address vanishing gradients
- **Memory Decoupling**: Separate short-term and long-term dependencies

These models capture temporal dynamics through recurrent connections but suffer from vanishing gradients and difficulty modeling long-range dependencies—limitations addressed by transformers.

### Spatial World Models: 3D and 4D Representations

For embodied AI applications like autonomous driving and robotics, modeling in 3D space is crucial. This has led to **occupancy-centric world models**.

**3D Occupancy Representation**: Instead of predicting 2D images, predict 3D voxel grids where each voxel indicates:
- Whether it's occupied
- Semantic class (car, pedestrian, road, etc.)
- Potentially velocity and other attributes

**4D Occupancy Forecasting**: Predict how 3D occupancy evolves over time, enabling:
- Geometry-aware perception
- Long-term scene forecasting
- Collision avoidance planning

**Recent Developments**:

**OccWorld (2024)**: Formulated 4D occupancy forecasting as learning to predict future 3D occupancy from past frames, using occupancy as a richer scene representation than bounding boxes.

**GaussianWorld (2026)**: Uses explicit 3D Gaussians (from 3D Gaussian Splatting) for scene representation instead of implicit voxels, enabling:
- Continuous, differentiable scene representation
- Explicit object movement modeling
- Streaming prediction for autonomous driving

**SparseWorld (2025)**: Employs sparse and dynamic queries for flexible, adaptive, and efficient 4D occupancy modeling, addressing computational costs of dense voxel representations.

**Key Advantages of Spatial World Models**:
1. **Geometry consistency**: Maintain 3D structure across viewpoints
2. **Multi-view support**: Predict multiple camera views consistently
3. **Action-conditioned prediction**: Simulate effects of ego-vehicle maneuvers
4. **Occlusion handling**: Model hidden spaces and object permanence

## Large-Scale Foundation World Models (2024-2026)

Perhaps the most exciting recent development is the emergence of **foundation world models**—large-scale models trained on diverse internet video that exhibit emergent world understanding.

### Google Genie Series

**Genie 1 (February 2024)**: A groundbreaking "foundation world model" that can generate playable, action-controllable virtual worlds from single images, photographs, or sketches—without requiring action labels during training.

**Architecture**:
- Spatiotemporal transformer for video generation
- Latent action model that discovers action space from video
- VQ-VAE for video tokenization

**Training**: 200K hours of internet gameplay videos, learning to infer latent actions from frame transitions.

**Genie 2 (December 2024)**: Scaled up significantly with improved consistency and interactivity.

**Genie 3 (August 2025)**: The first **real-time interactive world model**, generating navigable 3D environments at 24 FPS in 720p resolution. Unlike previous systems requiring explicit 3D representations, Genie 3 learns physics purely from observation—a major milestone showing emergent understanding of 3D space and dynamics from 2D video.

### OpenAI Sora

**Sora (February 2024)**: Demonstrated that scaling video generation models with diffusion transformers produces remarkable physical understanding. Sora can generate up to one minute of high-fidelity video with complex camera motion, multiple characters, and physics-based interactions.

**Key Insight**: OpenAI's research suggests that "video generation models are emerging as a promising path toward building general purpose simulators of the physical world."

**Architecture**:
- Diffusion transformer trained on video patches
- Variable duration, resolution, and aspect ratio support
- Text and image conditioning

**Emergent Capabilities**: Sora exhibits surprising understanding of:
- Object permanence (objects exist when occluded)
- 3D consistency across viewpoint changes
- Physical interactions and collisions
- Cause and effect relationships

**Limitations**: Sora sometimes makes physics errors (e.g., glass not shattering when hit), highlighting that current world models still imperfectly capture physical laws.

**Sora 2 (2025)**: Improved physics modeling, added audio generation, and multi-scene control with consumer access.

### Meta V-JEPA

**Joint-Embedding Predictive Architecture (JEPA)**: Yann LeCun's proposed architecture for learning world models by predicting representations rather than raw pixels.

**Key Idea**: Instead of reconstructing detailed pixel values (which wastes capacity on irrelevant details), predict abstract representations of future states:

$$
\text{Encoder}(x_{t+1}) \approx \text{Predictor}(\text{Encoder}(x_t), a_t)
$$

**V-JEPA (2024)**: Learns by predicting masked regions of video in representation space, excelling at understanding detailed object interactions without generating pixels.

**V-JEPA 2 (January 2026)**: Achieved state-of-the-art results after training on 1M+ hours of internet video, with 65-80% success on robotics pick-and-place tasks using only 62 hours of robot training data—demonstrating powerful transfer from video to physical interaction.

**Advantages of JEPA**:
- Energy efficient (no pixel generation)
- Focuses on semantic understanding
- Natural uncertainty representation in latent space
- Avoids common failure modes of generative models (e.g., hallucination)

### NVIDIA Cosmos

**Cosmos 1.0 (2025)**: NVIDIA's open-weight world foundation model platform focused on physical AI, offering models and benchmarks for:
- Autonomous driving
- Robot manipulation
- 3D consistency
- Physical alignment

**Key Features**:
- Generates realistic physics-based training data
- Supports multiple sensor modalities (camera, lidar, radar)
- Efficient video tokenization and generation
- Integration with simulation environments

**Purpose**: Enable cost-effective training and testing of autonomous systems by generating vast amounts of realistic synthetic data.

### World Labs

Fei-Fei Li's World Labs (2024-2025) recently launched **Marble**, a system to create 3D worlds from text, images, video, or coarse 3D layouts—focusing on spatial intelligence and scene understanding.

### The Scaling Hypothesis

These foundation models support a key hypothesis: **world models exhibit favorable scaling properties**. Just as LLMs improved dramatically with scale, world models show emergent capabilities when trained on diverse video data with sufficient compute:

- Better physics understanding
- Improved long-term consistency
- Generalization to novel scenarios
- Action-conditioned generation

However, scaling world models faces unique challenges compared to language:
- Video data is higher bandwidth than text
- Physics understanding requires 3D reasoning
- Evaluation is more complex than language metrics

## The Dreamer Series: State-of-the-Art MBRL

While foundation world models focus on large-scale video generation, the Dreamer series represents the cutting edge of world models for reinforcement learning.

### DreamerV1 (2020)

Introduced the **Recurrent State Space Model (RSSM)** that learns both deterministic and stochastic latent dynamics:

**Representation model** (encode observations):
$$
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
$$
$$
z_t \sim q(z_t | h_t, o_t)
$$

**Transition model** (predict dynamics):
$$
\hat{z}_t \sim p(z_t | h_t)
$$

**Observation model** (decode to observations):
$$
\hat{o}_t \sim p(o_t | h_t, z_t)
$$

**Reward model**:
$$
\hat{r}_t \sim p(r_t | h_t, z_t)
$$

**Key innovation**: Train actor and critic networks purely on imagined rollouts in latent space, using gradients from the learned world model.

### DreamerV2 (2021)

Improved upon V1 with:
- **Categorical latent states**: Replace Gaussian latents with discrete distributions
- **KL balancing**: Better trade-off between reconstruction and dynamics prediction
- **Simplified architecture**: Single world model for all tasks

Achieved human-level performance on Atari from pixels using only 2 hours of gameplay—remarkable sample efficiency.

### DreamerV3 (2023)

The current state-of-the-art, DreamerV3 is a **general-purpose algorithm** that masters diverse control tasks from a single configuration:
- Atari games
- DMC continuous control
- Minecraft (first agent to collect diamonds from scratch without human data!)
- Robotic manipulation

**Key innovations**:
1. **Symlog predictions**: Robust value and reward prediction via symmetric logarithm transformations
2. **Improved normalization**: EMA-based normalization for stability across scales
3. **Automatic entropy tuning**: Adaptive exploration without manual tuning

**Results**: Outperforms specialized algorithms on their respective domains while using a single hyperparameter configuration—a major milestone toward general-purpose MBRL.

### Dreamer4 (2025)

**Scalable World Model Training**: Dreamer4 extends the framework to much larger scales:
- Fast, accurate world model with efficient transformer architecture
- **Real-time interactive inference** on a single GPU
- Trained on offline data without environment interaction

**Minecraft Achievement**: First agent to obtain diamonds in Minecraft purely from offline data, outperforming OpenAI's VPT by 100x in data efficiency.

The Dreamer series demonstrates that world models can serve as powerful foundations for reinforcement learning across diverse domains.

## Applications: Where World Models Make Impact

### Autonomous Driving

World models have become central to autonomous vehicle development in 2025-2026:

**Waymo World Model (February 2026)**: Built on Google DeepMind's Genie 3, Waymo's world model generates hyper-realistic driving simulations with camera and lidar data for training on rare, dangerous scenarios:
- Wrong-way drivers
- Objects falling from trucks
- Flooded roads
- Unusual pedestrian behavior

**Key Benefits**:
- **Safety**: Test dangerous scenarios without risk
- **Coverage**: Generate edge cases rarely seen in real data
- **Efficiency**: Cheaper than real-world data collection
- **Iteration speed**: Rapidly test system changes

**Technical Approach**: Action-conditioned video generation where the AV's planned trajectory influences the predicted future scene, enabling counterfactual analysis ("What if I had braked harder?").

**Industry Adoption**: As of 2026, world models are recognized as a key enabling technology for autonomous driving, with Gartner listing physical AI as a top strategic technology trend.

### Robotics and Manipulation

World models enable robots to learn complex manipulation skills with far fewer real-world interactions:

**Simulation-to-Reality Transfer**: Train policies in simulated worlds, then transfer to real robots:
- Physics-informed world models learn from simulation
- Domain randomization improves robustness
- Fine-tuning with limited real data bridges the sim-to-real gap

**V-JEPA for Robotics**: Meta's V-JEPA 2 achieved 65-80% success on pick-and-place tasks with unfamiliar objects after training on only 62 hours of robot data—the world model pre-trained on internet video provided rich priors about object interactions.

**Contact-Rich Manipulation**: World models must capture:
- Contact dynamics and friction
- Deformable object behavior
- Multi-object interactions
- Occlusions and partial observability

**Open Challenges**: Robots require extremely high precision; small model errors can cause failure. Current research focuses on:
- Uncertainty quantification
- Physics-informed architectures
- Multi-modal world models (vision + force/torque sensing)

### Game Playing and Interactive Environments

World models have long been applied to game playing, with recent dramatic advances:

**MuZero (2020)**: DeepMind's MuZero learns to play games (Go, chess, shogi, Atari) at superhuman level without knowing the rules, learning a world model that predicts policy, value, and rewards without reconstructing observations.

**Minecraft World Models**:
- **MineWorld (2025)**: Real-time, open-source interactive world model driven by visual-action autoregressive transformers
- **Solaris (2026)**: Multiplayer world model with 12.64M frames, enabling consistent multi-view simulation and multiplayer interactions
- **Oasis**: Technical demo of real-time generative Minecraft gameplay

**Procedural Content Generation**: World models enable:
- Generating novel game levels
- Creating interactive NPCs with realistic behavior
- Dynamic difficulty adjustment

### Scientific Simulation

World models are increasingly applied to scientific domains:

**Weather and Climate**: World models for global weather prediction, learning atmospheric dynamics from historical data.

**Molecular Dynamics**: Predicting molecular conformations and interactions for drug discovery.

**Urban Planning**: UrbanDiT and similar models predict traffic flow, pedestrian movement, and urban dynamics for city planning.

**Physics Simulation**: Learning approximations to expensive physics simulations (fluid dynamics, structural mechanics) for faster engineering iteration.

## Technical Deep Dive: Training and Optimization

### Training Objectives

World models are typically trained to maximize likelihood of observed transitions:

$$
\max_\theta \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} [\log p_\theta(s_{t+1} | s_t, a_t)]
$$

**Reconstruction Loss**: For image-based observations:
$$
\mathcal{L}_{\text{recon}} = \mathbb{E}[\|o_{t+1} - \hat{o}_{t+1}\|^2]
$$

**KL Regularization**: For VAE-based models, regularize latent distributions:
$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q(z | o) \| p(z))
$$

**Reward Prediction**: For RL applications:
$$
\mathcal{L}_{\text{reward}} = \mathbb{E}[\|r_t - \hat{r}_t\|^2]
$$

**Total Objective**: Weighted combination:
$$
\mathcal{L}_{\text{world}} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + \gamma \mathcal{L}_{\text{reward}}
$$

### Handling Multi-Step Prediction

A key challenge: how to train models that can predict multiple steps into the future?

**Teacher Forcing**: Use ground-truth previous states during training:
$$
\hat{s}_{t+1} = f_\theta(s_t, a_t) \quad \text{(using true } s_t \text{)}
$$

This is efficient but creates train-test mismatch—at test time, the model uses its own predictions.

**Scheduled Sampling**: Gradually mix ground-truth and predicted states during training to reduce mismatch.

**Free Running**: Train on full rollouts using predicted states (no teacher forcing). This is more robust but harder to optimize due to compounding errors.

**Latent Overshooting** (used in Dreamer): Predict multiple steps ahead in latent space and supervise all predictions:
$$
\mathcal{L}_{\text{overshoot}} = \sum_{k=1}^K \|\text{Encode}(o_{t+k}) - \text{Predict}^k(z_t, a_{t:t+k})\|^2
$$

This regularizes the model to maintain accuracy over multiple steps.

### Dealing with Compounding Errors

**The Fundamental Challenge**: Small one-step prediction errors compound exponentially over long horizons:

If single-step error is $\epsilon$, then $H$-step error can grow as $\epsilon^H$ in worst case.

**Mitigation Strategies**:

1. **Latent Space Modeling**: Learn smooth latent representations where errors compound more slowly
2. **Hierarchical Modeling**: Predict at multiple temporal scales (high-level plans + low-level details)
3. **Error Correction**: Use model ensembles or uncertainty-aware planning to correct trajectory
4. **Hybrid Approaches**: Combine world model with model-free components
5. **Periodic Re-planning**: Re-plan frequently using fresh observations (Model Predictive Control)

**Recent Advances**: Research on "learning to combat compounding error" focuses on:
- Training objectives that explicitly penalize accumulated error
- Architectures that maintain long-term consistency
- Uncertainty estimation to detect when the model becomes unreliable

### Multi-Modal Dynamics and Mixture Models

Real-world dynamics are often multi-modal—multiple possible futures exist:

**Problem**: Single Gaussian predictions average over modes, producing blurry, unrealistic predictions.

**Solution**: Mixture Density Networks model the distribution as a mixture:
$$
p(s_{t+1} | s_t, a_t) = \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)
$$

where $\pi_k$, $\mu_k$, $\Sigma_k$ are predicted by the network.

**Alternatives**:
- **Implicit generative models**: VAEs, GANs, diffusion models naturally sample from multi-modal distributions
- **Categorical latents**: Dreamer uses categorical distributions in latent space to capture discrete variations

### Sample Efficiency and Data Requirements

**MBRL's Promise**: World models should enable learning from less data by:
1. Generating synthetic experience for policy learning
2. Transferring knowledge from pre-trained models
3. Enabling offline learning from static datasets

**Reality Check**: Current world models require:
- **Foundation models**: Millions to billions of video frames for pre-training
- **Task-specific models**: Thousands to millions of environment steps
- **Fine-tuning**: Often hundreds to thousands of additional samples

**Improving Sample Efficiency**:
- Pre-train on diverse data, fine-tune on task
- Model-based data augmentation
- Causal reasoning to focus on relevant variables
- Hierarchical and compositional world models

## Implementation Guide

### Popular Frameworks and Libraries

**Dreamer Implementations**:
```python
# DreamerV3 (official implementation)
# https://github.com/danijar/dreamerv3
import dreamerv3

config = dreamerv3.Config.from_yaml('configs/dreamerv3.yaml')
agent = dreamerv3.Agent(config)
agent.train(env)
```

**World Models (PyTorch)**:
```python
# Classical World Models implementation
# https://github.com/ctallec/world-models
from worldmodels import VAE, MDNRNN, Controller

# Train VAE
vae = VAE(latent_dim=32)
vae.train(observations)

# Train MDN-RNN
mdrnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256)
mdrnn.train(latents, actions, next_latents, rewards)

# Train controller with CMA-ES
controller = Controller(obs_dim=32, hidden_dim=256, action_dim=3)
controller.train(vae, mdrnn, env, generations=1000)
```

**Diffusion Models for Dynamics**:
```python
# Using diffusers library for video prediction
from diffusers import DDPMScheduler, UNet3DConditionModel

# Define 3D UNet for video
model = UNet3DConditionModel(
    sample_size=(64, 64),
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D"),
)

# Diffusion scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Training loop
for batch in dataloader:
    # Sample timestep and noise
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),))
    noise = torch.randn_like(batch)
    noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)

    # Predict noise
    pred_noise = model(noisy_batch, timesteps, conditioning).sample
    loss = F.mse_loss(pred_noise, noise)
    loss.backward()
```

### Building a Simple World Model

Let's implement a basic world model for a simple environment:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=256, hidden_dim=512):
        super().__init__()

        # Encoder: observations -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )

        # Dynamics: (latent_state, action) -> next_latent_state
        self.dynamics_rnn = nn.GRUCell(latent_dim + action_dim, latent_dim)

        # Decoder: latent state -> observations
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, obs):
        """Encode observation to latent distribution."""
        h = self.encoder(obs)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, latent):
        """Decode latent state to observation."""
        return self.decoder(latent)

    def predict_reward(self, latent):
        """Predict reward from latent state."""
        return self.reward_head(latent)

    def forward(self, obs, actions):
        """
        Full forward pass: encode, predict dynamics, decode.

        Args:
            obs: [batch_size, seq_len, obs_dim]
            actions: [batch_size, seq_len, action_dim]

        Returns:
            predictions: dict with reconstructed obs, predicted rewards, etc.
        """
        batch_size, seq_len = obs.shape[:2]

        # Encode initial observation
        mean, logvar = self.encode(obs[:, 0])
        latent = self.reparameterize(mean, logvar)
        hidden = latent  # Initial RNN hidden state

        predictions = {
            'obs': [],
            'rewards': [],
            'latents': [],
            'kl_loss': []
        }

        # Rollout through sequence
        for t in range(seq_len):
            # Predict reward
            reward = self.predict_reward(latent)
            predictions['rewards'].append(reward)

            # Decode to observation
            obs_pred = self.decode(latent)
            predictions['obs'].append(obs_pred)
            predictions['latents'].append(latent)

            # KL divergence (regularization)
            kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
            predictions['kl_loss'].append(kl)

            # Update latent state with action via dynamics model
            if t < seq_len - 1:
                action = actions[:, t]
                dynamics_input = torch.cat([latent, action], dim=-1)
                hidden = self.dynamics_rnn(dynamics_input, hidden)

                # Encode next observation for teacher forcing
                mean, logvar = self.encode(obs[:, t + 1])
                latent = self.reparameterize(mean, logvar)

        # Stack predictions
        predictions['obs'] = torch.stack(predictions['obs'], dim=1)
        predictions['rewards'] = torch.stack(predictions['rewards'], dim=1)
        predictions['latents'] = torch.stack(predictions['latents'], dim=1)
        predictions['kl_loss'] = torch.stack(predictions['kl_loss'], dim=1)

        return predictions

    def imagine(self, initial_obs, actions):
        """
        Generate imagined rollout without teacher forcing.

        Args:
            initial_obs: [batch_size, obs_dim]
            actions: [batch_size, horizon, action_dim]

        Returns:
            imagined trajectory
        """
        batch_size, horizon = actions.shape[:2]

        # Encode initial state
        mean, logvar = self.encode(initial_obs)
        latent = self.reparameterize(mean, logvar)
        hidden = latent

        imagined = {
            'obs': [],
            'rewards': [],
            'latents': []
        }

        # Free running imagination
        for t in range(horizon):
            # Predict reward
            reward = self.predict_reward(latent)
            imagined['rewards'].append(reward)

            # Decode observation
            obs_pred = self.decode(latent)
            imagined['obs'].append(obs_pred)
            imagined['latents'].append(latent)

            # Dynamics step
            action = actions[:, t]
            dynamics_input = torch.cat([latent, action], dim=-1)
            hidden = self.dynamics_rnn(dynamics_input, hidden)
            latent = hidden  # No encoding in imagination

        imagined['obs'] = torch.stack(imagined['obs'], dim=1)
        imagined['rewards'] = torch.stack(imagined['rewards'], dim=1)
        imagined['latents'] = torch.stack(imagined['latents'], dim=1)

        return imagined

def train_world_model(world_model, dataloader, epochs=100, beta_kl=0.1):
    """Train the world model on collected data."""
    optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            obs, actions, rewards = batch

            # Forward pass
            predictions = world_model(obs, actions)

            # Losses
            obs_loss = F.mse_loss(predictions['obs'], obs)
            reward_loss = F.mse_loss(predictions['rewards'], rewards.unsqueeze(-1))
            kl_loss = predictions['kl_loss'].mean()

            loss = obs_loss + reward_loss + beta_kl * kl_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

# Example usage
obs_dim = 20  # e.g., state vector for CartPole
action_dim = 1  # discrete action encoded as continuous
world_model = SimpleWorldModel(obs_dim, action_dim)

# Training would proceed with collected environment data
# train_world_model(world_model, dataloader)
```

### Training Tips

**1. Start Simple**: Begin with simple environments (CartPole, Pendulum) before tackling complex visual domains.

**2. Balance Reconstruction and Dynamics**: Too much emphasis on reconstruction can hurt dynamics prediction. The $\beta$ parameter in VAE loss is crucial.

**3. Monitor Multi-Step Accuracy**: Track prediction error at horizons 1, 5, 10+ steps to catch compounding error issues.

**4. Use Latent Overshooting**: Regularize long-term predictions even during training.

**5. Curriculum Learning**: Gradually increase prediction horizon during training.

**6. Ensemble Models**: Train multiple world models and use disagreement for uncertainty estimation.

**7. Data Diversity**: Ensure training data covers the full state-action space; world models struggle to extrapolate.

## Challenges and Open Problems

Despite recent progress, many fundamental challenges remain:

### Compounding Errors

**The Problem**: Small prediction errors accumulate exponentially over long horizons. After $H$ steps, even a 1% per-step error can make predictions useless.

**Current Approaches**:
- Periodic re-planning with fresh observations (MPC)
- Hierarchical models that avoid fine-grained long-term prediction
- Uncertainty-aware planning that detects unreliable regions
- Hybrid model-free + model-based methods

**Open Questions**: Can we develop world models with provable bounds on error accumulation? How should we trade off model fidelity vs. planning horizon?

### Sample Efficiency

**The Paradox**: Model-based RL should be sample efficient, but current world models often require massive amounts of data to train:
- Foundation models: billions of video frames
- Task-specific models: millions of environment steps

**Factors**:
- High-dimensional observations (images, video)
- Stochastic, complex dynamics
- Distribution shift between training and deployment

**Promising Directions**:
- Transfer learning from pre-trained models
- Physics-informed architectures that build in inductive biases
- Active learning and curiosity-driven exploration
- Compositional world models that generalize to novel combinations

### Physical Consistency

**Challenge**: Current world models frequently violate physical laws:
- Objects passing through each other
- Energy not conserved
- Impossible trajectories
- Inconsistent shadows and reflections

**Why It's Hard**: Learning physics from pixels without explicit supervision is extremely challenging. Models may achieve visual plausibility without true physical understanding.

**Approaches**:
- Physics-informed loss functions (energy conservation, momentum, etc.)
- Hybrid simulation + learning (use physics engines with learned parameters)
- Geometric and 3D-aware architectures
- Contrastive learning on physically possible vs. impossible scenarios

**Recent Work**: Physics-Informed BEV World Model (PIWM) and similar efforts show promise but are still far from perfect physical consistency.

### Long-Horizon Prediction

**Fundamental Tension**: The longer the prediction horizon, the more uncertainty and the harder the problem.

**Strategies**:
- **Hierarchical modeling**: Predict high-level "waypoints" and fill in details
- **Stochastic prediction**: Represent uncertainty rather than single trajectories
- **Goal-conditioned models**: Condition on desired end-state rather than predicting full trajectory
- **Diffusion for long-horizon**: Iteratively refine entire trajectories

**Open Problem**: How to maintain both accuracy and diversity in long-horizon predictions?

### Generalization

**Out-of-Distribution Challenge**: World models often fail on states, actions, or scenarios not well-represented in training data.

**Types of Generalization Needed**:
- Novel objects and appearances
- Different physical parameters (friction, mass, etc.)
- New tasks and goals
- Compositional generalization (new combinations of known elements)

**Approaches**:
- Domain randomization during training
- Causal models that separate stable from variable factors
- Meta-learning across environments
- Foundation models trained on diverse data

### Evaluation

**The Measurement Problem**: How do we properly evaluate world models?

**Metrics**:
- Pixel-level prediction error (MSE, SSIM, FVD)
- Perceptual similarity (LPIPS, CLIP similarity)
- Task performance (reward in RL)
- Physical consistency scores
- Human evaluation

**Issues**: Pixel metrics don't capture physical understanding; task metrics don't isolate model quality; human evaluation doesn't scale.

**Recent Benchmarks**:
- **WorldBench**: Tests concept-specific physical reasoning
- **WorldModelBench**: Judges world modeling across domains
- **WoW-World-Eval**: Comprehensive embodied AI evaluation

**Open Question**: What are the right metrics for "understanding" vs. "generation quality"?

### Computational Cost

**Reality Check**: Training and inference with world models is expensive:
- High-dimensional state spaces (video)
- Long sequences
- Iterative generation (diffusion models)
- Multi-step planning

**Efficiency Techniques**:
- Latent space modeling (lower dimensional than pixels)
- Sparse representations (occupancy grids, object-centric)
- Efficient architectures (linear attention, state-space models)
- Distillation (train large model, deploy small model)

**Trade-offs**: Accuracy vs. speed is always present; the right choice depends on application.

## Future Directions

Where is the field headed? Several exciting directions are emerging:

### Unified Multimodal World Models

Future world models will likely integrate multiple modalities:
- **Vision + audio**: Predict sound consistent with visual events
- **Vision + language**: Use language to guide and explain predictions
- **Vision + force/torque**: For robotic manipulation
- **Multi-sensor fusion**: Camera, lidar, radar, GPS, IMU

**Unified-IO 2** and similar models show the potential of joint training across modalities.

### Compositional and Object-Centric Models

Instead of monolithic pixel prediction, decompose scenes into objects and relationships:
- Predict object-level dynamics and interactions
- Enable combinatorial generalization
- Provide interpretable, editable representations

**Latent Particle World Model (LPWM)** is a recent example of this direction.

### Causal World Models

Explicitly model causal relationships:
- Identify stable causal structure vs. confounding factors
- Enable interventional reasoning ("What if I had acted differently?")
- Improve robustness and generalization

**Challenges**: Causal discovery from observational data is fundamentally hard; need inductive biases and possibly interventional data.

### Real-Time Interactive Models

Genie 3 demonstrated real-time world model inference, but most models are still too slow for interactive use:

**Requirements**:
- Sub-100ms latency for responsive interaction
- Continuous streaming (not batch processing)
- Efficient state updates (no full recomputation)

**Applications**: Video games, VR/AR, teleoperation, interactive training.

### World Models for Reasoning

Can world models serve as "mental simulators" for abstract reasoning?
- Simulate social interactions and theory of mind
- Mathematical and logical reasoning via simulation
- Creative problem-solving through mental search

This connects to broader questions about system 2 reasoning and cognitive architectures.

### Open-Source and Democratization

As with LLMs, open-source world models are crucial for research progress:
- Open weights and architectures
- Standard benchmarks and evaluation
- Efficient implementations for smaller labs
- Ethical guidelines and safety considerations

NVIDIA's Cosmos and various academic efforts are moving in this direction.

## Conclusion

World models represent a fundamental capability for intelligent systems: the ability to predict, simulate, and reason about how the world evolves over time. From their roots in Kalman filtering and control theory to modern foundation models trained on billions of video frames, world models have come remarkably far.

The past few years have witnessed explosive progress:
- Transformers and diffusion models bringing unprecedented generation quality
- Foundation world models exhibiting emergent physical understanding
- Real-world deployments in autonomous driving and robotics
- Sample-efficient algorithms like DreamerV3 mastering diverse tasks

Yet fundamental challenges remain: compounding errors, physical consistency, computational cost, and robust evaluation. Solving these will require interdisciplinary insights from machine learning, control theory, physics, cognitive science, and more.

As we look toward the future, world models are poised to play a central role in the next generation of AI systems—ones that don't just process information, but truly understand and interact with the physical world. Whether in self-driving cars navigating city streets, robots manipulating objects in warehouses, or game agents exploring virtual worlds, the ability to predict and plan using learned world models will be essential.

The journey from Ha and Schmidhuber's pioneering work to today's real-time interactive foundation models has been remarkable. But in many ways, we're still in the early stages. As compute grows, algorithms improve, and data becomes more abundant, world models will become increasingly powerful, general, and indispensable—learning not just to dream, but to truly understand reality.

---

## Further Reading and Resources

### Seminal Papers

- Ha, D., & Schmidhuber, J. (2018). "World Models" ([arXiv:1803.10122](https://arxiv.org/abs/1803.10122))
- Hafner, D., et al. (2023). "Mastering Diverse Domains through World Models" ([arXiv:2301.04104](https://arxiv.org/abs/2301.04104))
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"

### Recent Work (2024-2026)

- Genie 1 & 2: Google DeepMind's foundation world models
- OpenAI Sora: Video generation models as world simulators ([OpenAI Blog](https://openai.com/index/video-generation-models-as-world-simulators/))
- V-JEPA: Meta's joint embedding predictive architecture ([Meta AI Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/))
- Waymo World Model: Real-world deployment for AV training ([Waymo Blog](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/))
- DreamerV4: Scalable world model agents ([arXiv:2509.24527](https://arxiv.org/abs/2509.24527))

### Surveys and Repositories

- [World Models GitHub Survey](https://github.com/tsinghua-fib-lab/World-Model): Comprehensive survey published in ACM CSUR 2025
- [Awesome World Models for Autonomous Driving](https://github.com/LMD0311/Awesome-World-Model)
- [From Video Generation to World Models](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model)
- [3D and 4D World Modeling Survey](https://worldbench.github.io/survey)

### Implementations

- [DreamerV3 Official Implementation](https://github.com/danijar/dreamerv3)
- [World Models PyTorch](https://github.com/ctallec/world-models)
- [Diffusion Transformers (DiT)](https://github.com/facebookresearch/DiT)

### Benchmarks

- [WorldBench](https://arxiv.org/abs/2601.21282): Diagnostic evaluation of world models
- [WorldModelBench](https://worldmodelbench.github.io/): Judging video generation as world models
- [EvalCrafter](https://evalcrafter.github.io/): Benchmarking video generation models

---

## Sources

This post synthesized information from numerous sources. Key references include:

- [Scientific American: World models could unlock the next revolution in AI](https://www.scientificamerican.com/article/world-models-could-unlock-the-next-revolution-in-artificial-intelligence/)
- [Euronews: What to expect from AI in 2026 - World Models](https://www.euronews.com/next/2026/01/01/from-ai-slop-to-world-models-bubbles-and-small-models-what-to-expect-from-ai-in-2026)
- [Original World Models Paper](https://arxiv.org/abs/1803.10122) by Ha & Schmidhuber
- [OpenAI: Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Meta AI: V-JEPA - The next step toward advanced machine intelligence](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [Waymo: The Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)
- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)
- [Medium: World Models Reading List 2025](https://medium.com/@graison/world-models-reading-list-the-papers-you-actually-need-in-2025-882f02d758a9)
- [Kalman Filter Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
- [Model-Based RL Resources](https://www.geeksforgeeks.org/artificial-intelligence/model-based-reinforcement-learning-mbrl-in-ai/)
- [3D and 4D World Modeling Survey](https://worldbench.github.io/survey)
- [Building Physically Plausible World Models](https://physical-world-modeling.github.io/)
- [Diffusion Transformers for Open-World Learning](https://arxiv.org/abs/2411.12164)
- Multiple arXiv preprints and conference papers from NeurIPS, ICLR, CVPR, ICML 2024-2026

