---
layout: post
title: "From Points to Pixels: A Deep Dive into 3D Gaussian Splatting"
date: 2026-03-20 10:00:00-0800
description: A comprehensive guide to understanding 3D Gaussian Splatting, from first principles to state-of-the-art applications in 2026
tags: computer-vision 3d-reconstruction neural-rendering gaussian-splatting
---

## Introduction

Imagine capturing a real-world scene with just a smartphone camera and, within minutes, reconstructing it as a fully interactive 3D environment that renders at over 100 frames per second with photorealistic quality. This isn't science fiction—it's the reality enabled by **3D Gaussian Splatting** (3DGS), a technique that has revolutionized 3D scene reconstruction and rendering since its introduction at SIGGRAPH 2023.

In this comprehensive guide, we'll journey from the fundamental concepts underlying Gaussian Splatting to its cutting-edge applications in AR/VR, gaming, and digital twins. Whether you're an AI researcher looking to understand the mathematical foundations or a practitioner seeking to apply these techniques, this post will build your intuition progressively while maintaining technical rigor.

## The Evolution: From NeRF to Gaussian Splatting

### Neural Radiance Fields: The Foundation

To understand Gaussian Splatting, we must first appreciate what came before it. Neural Radiance Fields (NeRF), introduced in 2020, represented a paradigm shift in 3D reconstruction. NeRF models a scene as a continuous 5D function that maps 3D spatial coordinates $(x, y, z)$ and viewing directions $(\theta, \phi)$ to color and density values:

$$F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

where $\mathbf{x} \in \mathbb{R}^3$ is the position, $\mathbf{d} \in \mathbb{S}^2$ is the viewing direction, $\mathbf{c} \in \mathbb{R}^3$ is the RGB color, and $\sigma \in \mathbb{R}^+$ is the volume density.

NeRF's volumetric rendering produces stunning photorealistic results by integrating along camera rays:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

where $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$ is the accumulated transmittance.

### The NeRF Bottleneck

Despite its revolutionary quality, NeRF had critical limitations:

1. **Slow Training**: Hours or even days to optimize for a single scene
2. **Slow Rendering**: Requires querying the neural network hundreds of times per ray (typically 64-128 samples)
3. **Real-time Impossibility**: Rendering at interactive frame rates (>30 FPS) was practically impossible
4. **Memory Intensive**: Volumetric integration is computationally expensive

This is where Gaussian Splatting enters the picture.

## What is Gaussian Splatting?

### The Core Idea

3D Gaussian Splatting represents a scene not as a continuous implicit function, but as a **collection of oriented 3D Gaussian distributions** in space. Think of it as representing your scene with thousands or millions of fuzzy, colored ellipsoids floating in 3D space.

Each Gaussian primitive is defined by:

- **Position** $\boldsymbol{\mu} \in \mathbb{R}^3$: The center of the Gaussian in 3D space
- **Covariance** $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$: Defining the shape and orientation (an ellipsoid)
- **Opacity** $\alpha \in [0, 1]$: How transparent the Gaussian is
- **Color** represented via spherical harmonics coefficients

The key insight: instead of ray marching through a continuous field, we **project and rasterize** these Gaussians onto the image plane—a process GPUs are exceptionally fast at.

### Mathematical Foundation

A 3D Gaussian is defined by the probability density function:

$$G(\mathbf{x}) = \frac{1}{(2\pi)^{3/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

To ensure the covariance matrix $\boldsymbol{\Sigma}$ remains symmetric and positive semi-definite (PSD), it's parameterized as:

$$\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

where:
- $\mathbf{S} \in \mathbb{R}^{3 \times 3}$ is a diagonal matrix of scale values
- $\mathbf{R} \in SO(3)$ is a rotation matrix constructed from a unit quaternion $\mathbf{q}$

This parameterization is crucial because:
1. It guarantees valid covariance matrices during optimization
2. Quaternions provide smooth, singularity-free rotation representation
3. It's fully differentiable with respect to scale and rotation parameters

## The Splatting Process: From 3D to 2D

### Projection to Image Space

The magic of splatting happens when we project 3D Gaussians to 2D. Given a camera with viewing transformation $\mathbf{W}$ and projection matrix $\mathbf{J}$, we project the 3D Gaussian's mean and covariance:

**Mean Projection:**
$$\boldsymbol{\mu}' = \mathbf{J} \mathbf{W} \boldsymbol{\mu}$$

**Covariance Projection (using the Jacobian):**
$$\boldsymbol{\Sigma}' = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^T \mathbf{J}^T$$

This derivation comes from EWA (Elliptical Weighted Average) splatting, a classical technique from texture filtering. The critical property: **this projection is fully differentiable**, enabling end-to-end optimization via backpropagation.

The 2D Gaussian on the image plane is:

$$G_{2D}(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}')^T {\boldsymbol{\Sigma}'}^{-1} (\mathbf{x} - \boldsymbol{\mu}')\right)$$

### View-Dependent Color with Spherical Harmonics

To capture view-dependent effects (like specular highlights), each Gaussian stores color using **spherical harmonic (SH) coefficients** rather than a single RGB value.

Spherical harmonics form an orthonormal basis on the sphere, analogous to how Fourier series work for periodic functions. The color for viewing direction $\mathbf{d}$ is:

$$\mathbf{c}(\mathbf{d}) = \sum_{l=0}^{l_{\text{max}}} \sum_{m=-l}^{l} c_{lm} Y_l^m(\mathbf{d})$$

where $Y_l^m$ are the spherical harmonic basis functions and $c_{lm}$ are learned coefficients (stored per channel: R, G, B).

In practice, 3DGS uses $l_{\text{max}} = 3$ (16 coefficients total), providing a good balance between expressiveness and parameter count. The view direction is computed from the camera center to the Gaussian center, and the SH basis functions are evaluated and combined with the learned coefficients to produce the final color.

**Recent Innovation**: Some methods now use **dual spherical harmonics** to separately model diffuse and specular components, or **Spherical Gaussians** to reduce parameters while maintaining quality.

### Tile-Based Rasterization

Rendering is performed using a custom **differentiable tile-based rasterizer**:

1. **Frustum Culling**: Discard Gaussians outside the camera frustum
2. **Tile Assignment**: Divide the image into 16×16 pixel tiles; assign each Gaussian to tiles it overlaps
3. **Sorting**: Within each tile, sort Gaussians by depth (front-to-back)
4. **Alpha Blending**: For each pixel, blend Gaussians using standard alpha compositing:

$$C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

where $c_i$ is the color and $\alpha_i = \alpha \cdot G_{2D}(\mathbf{x}_{\text{pixel}})$ is the opacity modulated by the 2D Gaussian falloff.

This process leverages GPU hardware optimizations, achieving **real-time performance** (100+ FPS at 1080p).

## Training: Optimization from Images

### Initialization from Structure-from-Motion

The optimization process begins with **Structure-from-Motion (SfM)**, typically using COLMAP, which provides:
- Sparse 3D point cloud
- Camera poses and intrinsics for each input image

Each SfM point initializes a 3D Gaussian:
- Position $\boldsymbol{\mu}$ = point position
- Covariance initialized as small isotropic (spherical)
- Color from the point cloud color
- Opacity initialized to a small value

**Dependency Challenge**: Standard 3DGS relies heavily on accurate SfM initialization—random initialization typically fails with 4-5 dB PSNR drops. Recent research explores alternatives like monocular depth estimation or feed-forward models to eliminate this dependency.

### Loss Function

The training objective is simple: photometric consistency with input views. The loss is:

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

where:
- $\mathcal{L}_1$ is the pixel-wise L1 loss
- $\mathcal{L}_{\text{D-SSIM}}$ is the structural dissimilarity (complement of SSIM)
- $\lambda = 0.2$ balances the two terms

This combination captures both pixel accuracy (L1) and perceptual quality (SSIM).

### Adaptive Density Control

A critical innovation in 3DGS is **Adaptive Density Control (ADC)**—dynamically adjusting the number and properties of Gaussians during training. Every 100 iterations (typically), the algorithm:

**1. Densification**: Add Gaussians in under-reconstructed regions
   - **Clone**: Replicate Gaussians with high positional gradients but small size
   - **Split**: Divide Gaussians with high gradients and large size into two smaller ones

**2. Pruning**: Remove unnecessary Gaussians
   - Remove Gaussians with opacity below a threshold (e.g., $\alpha < 0.005$)
   - Remove extremely large Gaussians that may be artifacts

**Gradient Threshold**: If the accumulated positional gradient over the last $\tau$ iterations exceeds a threshold, the Gaussian is a candidate for densification.

**Recent Improvements**: The original clone/split operations have known limitations—they can produce many overlapping, low-opacity Gaussians. Recent work proposes:
- **Long-axis split**: Split along the major axis to reduce overlap
- **Opacity-adaptive pruning**: More sophisticated removal strategies
- **Gaussian importance weighting**: Prioritize meaningful Gaussians

The entire process is **fully differentiable**, allowing gradients to flow from the rendered image back to all Gaussian parameters through the projection and rasterization.

## Technical Deep Dive: Why Is It So Fast?

The speed of Gaussian Splatting comes from multiple factors:

### 1. Explicit Representation
Unlike NeRF's implicit neural network, Gaussians are **explicit primitives**—no expensive neural network evaluation per sample.

### 2. GPU-Optimized Rasterization
The tile-based rasterizer is designed for modern GPU architectures:
- Coalesced memory access patterns
- Minimal divergence within warps
- Hardware-accelerated alpha blending

### 3. Efficient Sorting
Per-tile sorting is performed on a relatively small number of Gaussians (compared to global sorting), and modern GPUs excel at parallel sorting.

### 4. Early Termination
Alpha compositing stops when accumulated opacity reaches $\sim 1$, avoiding processing far-away Gaussians.

### Quantitative Comparison

| Method | Training Time | Rendering Speed | Quality (PSNR) | Memory |
|--------|--------------|-----------------|----------------|---------|
| NeRF (original) | 24-48 hours | 0.1 FPS | High | Low |
| Instant-NGP | 5-10 minutes | 1-10 FPS | High | Medium |
| 3D Gaussian Splatting | 10-30 minutes | 100+ FPS | High | Medium-High |

## Gaussian Splatting vs. NeRF: A Nuanced Comparison

While 3DGS has captured significant attention, it's essential to understand the tradeoffs:

### Gaussian Splatting Advantages
- **Real-time rendering** (100+ FPS vs. <1 FPS)
- **Faster training** (minutes vs. hours)
- **Deterministic rendering** (no stochastic ray sampling)
- **Better for dynamic scenes** with explicit representations

### NeRF Advantages
- **Better generalization** to novel views outside training distribution
- **More compact representation** (neural weights vs. millions of Gaussians)
- **Smoother geometry** in low-texture regions
- **More stable training** with limited views
- **Better with exposure variation** and motion blur

### Use Case Recommendations
- **3DGS**: Real-time applications (VR/AR, gaming, live simulations), interactive visualization, production rendering
- **NeRF**: High-quality offline rendering, scenes with sparse input views, memory-constrained environments (robotics)

## State-of-the-Art in 2026

As of March 2026, Gaussian Splatting has evolved from a research curiosity to a production-ready technology. Here are the major developments:

### Industry Adoption

**Major Software Integration:**
- Foundry's **Nuke 17.0** officially added Gaussian Splatting support
- **OpenUSD 26.03** includes native Gaussian Splatting representation
- **VRChat** added Gaussian Splatting support for user-created worlds
- Khronos Group released a **glTF 2.0 release candidate** with Gaussian Splatting extensions

**Hollywood Production**: The 2026 Superman film marked the first major motion picture using **dynamic Gaussian Splatting** (4DGS) in production, demonstrating the technology's maturity for professional VFX workflows.

**Live Events**: Gaussian Splatting was deployed for Olympic coverage in Milan 2026, providing immersive views of ski jumping, hockey, and figure skating.

### 4D Gaussian Splatting: Dynamic Scenes

A major research direction is extending 3DGS to **dynamic scenes**—scenes that change over time. This is often called **4D Gaussian Splatting** (4DGS), where time is the fourth dimension.

**Key Approaches:**

1. **Explicit Time Encoding**: Each Gaussian has time-varying properties encoded via neural voxels or deformation fields

2. **Native 4D Primitives**: Instead of 3D Gaussians deformed over time, directly optimize 4D Gaussian primitives with explicit geometry and appearance in spacetime

**Major Publications:**
- **CVPR 2024**: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering" achieved 82 FPS at 800×800 on RTX 3090
- **SIGGRAPH 2024**: "ST-4DGS" with spatial-temporal consistency and motion-aware shape regularization
- **ICCV 2025**: "MEGA" reduced storage from 13M to 0.91M Gaussians for complex scenes
- **NeurIPS 2025**: "4DGS-1K" running at 1000+ FPS with 41× storage reduction

**Commercial Deployment**: Gracia AI raised $1.7M for an end-to-end 4DGS platform with Unity/Unreal plugins and standalone playback on Meta Quest 3/3S and Pico 4 Ultra.

### VR/AR Optimization

**VRSplat**: The first systematically evaluated 3DGS for VR, achieving 72+ FPS while eliminating:
- Temporal artifacts (popping during head movement)
- Stereo-disrupting floaters
- View-inconsistent projections

**Foveated Rendering**: Combining 3DGS with gaze tracking for dynamic resolution adjustment, achieving 90Hz at full VR resolution (2016×2240 per eye).

**Photorealistic Avatars**: End-to-end pipelines creating Gaussian Splatting avatars from monocular video, directly compatible with Unity, with rendering speeds up to 361 FPS.

### Compression and Efficiency

**Challenge**: A complex scene can require 10-13 million Gaussians (~7-8 GB storage), limiting deployment on mobile and web platforms.

**Solutions:**
- **Feed-forward compression** with long-context modeling achieving 20× compression ratios
- **Pruning and quantization** reducing parameters while maintaining quality
- **Learned codecs** specifically designed for Gaussian primitives

### Enhanced Quality

**Depth Integration**: Methods like **DepthSplat** incorporate monocular depth estimation (e.g., from MiDaS or Depth Anything) as geometric priors, improving:
- Reconstruction in low-texture regions
- Geometric accuracy
- Performance with fewer input views

**Planar Priors**: For indoor scenes, detecting planes (walls, floors) and constraining Gaussians improves texture-complexity-aware initialization.

**2D Gaussians**: SIGGRAPH 2024 introduced **2D Gaussian Splatting**, representing surfaces as oriented 2D disks rather than 3D ellipsoids, improving geometric accuracy.

### Human Reconstruction

Specialized methods for digitizing humans:
- **SiTH, PSHuman**: Integrate diffusion models to infer occluded body parts
- **PARTE**: Enhanced geometric detail for clothing and hair
- **GSAC**: Unity integration for avatar creation from video

**Open Challenges**:
- Realistic cloth dynamics and loose garments
- Few-shot learning (reconstruction from 1-5 images)
- Extracting editable rigged meshes for animation

## Applications Across Industries

### Gaming and Virtual Production
- **Real-time game environments** from photogrammetry scans
- **Previs and techvis** for film production (Superman 2026)
- **In-camera VFX (ICVFX)** with volumetric backgrounds that can be re-lit in real-time

### AR/VR and Spatial Computing
- **Digital twins** of real spaces for training and simulation
- **Telepresence** with photorealistic human avatars
- **Location-based VR** experiences (theme parks, museums)

### Geospatial and UAV Mapping
- **Drone photogrammetry** reconstructions combining 3DGS with LiDAR
- **Urban digital twins** for city planning and infrastructure
- **Environmental monitoring** with temporal 4DGS

### Medical and Scientific Visualization
- **Anatomical reconstruction** from CT/MRI scans
- **Microscopy data** visualization in 3D
- **Surgical planning** with patient-specific 3D models

### E-commerce and Retail
- **Product visualization** with 360° view-dependent rendering
- **Virtual try-on** for furniture and fashion
- **Showroom digitization** for online shopping experiences

## Implementation: Getting Started

### Popular Repositories

**Official Implementation:**
- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) - The reference SIGGRAPH 2023 implementation

**Optimized Alternatives:**
- [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) - 4× less memory, 15% faster than official, with extensive new features
- [gaussian-splatting-lightning](https://github.com/yzslab/gaussian-splatting-lightning) - Unified framework with multiple algorithm implementations and web viewer
- [GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio) - Modular framework with various paper implementations

**Web and Viewer:**
- [mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) - Three.js-based web viewer for interactive visualization

**Curated Resources:**
- [MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) - Comprehensive list of papers, implementations, and resources

### Basic Workflow

```python
# Pseudocode for 3D Gaussian Splatting training loop

# 1. Initialize from SfM point cloud
gaussians = initialize_from_sfm(colmap_output)

# 2. Optimization loop
for iteration in range(num_iterations):
    # Sample a training view
    camera, gt_image = sample_training_view()

    # Render the scene
    rendered_image = rasterize_gaussians(gaussians, camera)

    # Compute loss
    loss = (1 - λ) * L1(rendered_image, gt_image) + \
           λ * DSSIM(rendered_image, gt_image)

    # Backpropagate
    loss.backward()

    # Update Gaussian parameters
    optimizer.step()

    # Adaptive density control every 100 iterations
    if iteration % 100 == 0:
        # Densification
        gaussians = clone_and_split(gaussians, gradient_threshold)

        # Pruning
        gaussians = prune_low_opacity(gaussians, α_threshold)
        gaussians = prune_large_gaussians(gaussians, size_threshold)

# 3. Save optimized Gaussians
save_ply(gaussians, "scene.ply")
```

### Training Your Own Scene

**Using NeRFStudio's gsplat:**

```bash
# Install NeRFStudio
pip install nerfstudio

# Process your video/images with COLMAP
ns-process-data images --data data/my_scene --output-dir data/my_scene_processed

# Train Gaussian Splatting
ns-train splatfacto --data data/my_scene_processed

# Interactive viewer
ns-viewer --load-config outputs/.../config.yml
```

**Key Hyperparameters:**
- **Learning rates**: Position (1.6e-4), color (2.5e-3), opacity (5e-2), scale (5e-3), rotation (1e-3)
- **Densification interval**: 100 iterations (from iteration 500 to 15,000)
- **Opacity reset**: Every 3,000 iterations to prevent over-saturation
- **Number of iterations**: 30,000 for typical scenes

## Open Problems and Future Directions

Despite remarkable progress, several challenges remain:

### 1. Compression and Memory Efficiency
**Problem**: Scenes require millions of Gaussians (5-10 GB), limiting mobile deployment.

**Directions**:
- Learned compression codecs
- Hierarchical representations
- Neural implicit-explicit hybrids

### 2. Reliance on Accurate Initialization
**Problem**: Poor SfM point clouds → poor reconstructions (4-5 dB PSNR drop).

**Directions**:
- Feed-forward models eliminating SfM dependency
- Self-supervised initialization from scratch
- Integration with foundation models (SAM, Depth Anything)

### 3. Geometric Accuracy
**Problem**: 3DGS optimizes for visual appearance, not geometric correctness—surfaces may be poorly defined.

**Directions**:
- 2D Gaussian Splatting for surface-aligned primitives
- Regularization encouraging surface consistency
- Hybrid mesh-Gaussian representations

### 4. Dynamic Scenes and Deformation
**Problem**: 4DGS storage and temporal consistency challenges.

**Directions**:
- More compact motion representations
- Physics-aware deformation models (cloth, fluid)
- Temporal super-resolution

### 5. Sparse View Reconstruction
**Problem**: Artifacts, scale ambiguity, multi-view inconsistency with few images.

**Directions**:
- Few-shot learning with diffusion priors
- Semantic guidance for plausible reconstruction
- Test-time optimization with pre-trained models

### 6. Standardization and Tooling
**Problem**: Lack of universal format for Gaussian scenes.

**Directions**:
- OpenUSD and glTF extensions (in progress for 2026)
- Production-grade editing tools
- Integration with existing 3D pipelines (Blender, Maya)

### 7. Quality Assessment
**Problem**: Existing metrics (PSNR, SSIM, LPIPS) don't fully capture subjective quality.

**Directions**:
- Perceptual metrics for Gaussian-based rendering
- User studies for quality benchmarks
- Task-specific evaluation protocols

## Conclusion: The JPEG Moment for Spatial Computing

As we stand in March 2026, 3D Gaussian Splatting represents what some call a "JPEG moment for spatial computing"—a transformative technology making photorealistic 3D capture and rendering accessible at scale. What began as an academic paper at SIGGRAPH 2023 has evolved into a production-ready technique deployed in Hollywood films, Olympic broadcasts, professional virtual production, and consumer VR/AR devices.

The technique's elegance lies in its simplicity: represent scenes with oriented 3D Gaussians, project them to 2D, and rasterize efficiently on GPUs. This explicit representation sidesteps the computational bottlenecks of implicit neural methods while maintaining photorealistic quality.

Yet, as we've explored, Gaussian Splatting isn't a silver bullet—it complements rather than replaces techniques like NeRF. The choice between methods depends on your constraints: real-time requirements favor 3DGS, memory constraints favor NeRF, geometric accuracy may require hybrid approaches.

Looking forward, the research community is addressing compression, sparse-view reconstruction, dynamic scenes, and geometric accuracy. The industry is converging on standards (OpenUSD, glTF), and tool ecosystems are maturing. By 2027, we can expect Gaussian Splatting to be as ubiquitous in 3D content creation as JPEG is for 2D images.

For researchers and practitioners entering this space, the opportunity is immense. Whether you're building the next generation of VR experiences, digitizing cultural heritage, creating digital twins for industry, or pushing the boundaries of real-time rendering—3D Gaussian Splatting provides a powerful foundation.

The code is open, the papers are accessible, and the community is collaborative. Now is the perfect time to dive in.

---

## Further Reading and Resources

### Foundational Papers

1. **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (SIGGRAPH 2023)
   Kerbl, Kopanas, Leimkühler, Drettakis
   [https://arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)

2. **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** (ECCV 2020)
   Mildenhall et al.
   [https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)

3. **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** (CVPR 2024)
   Wu et al.
   [https://arxiv.org/abs/2310.08528](https://arxiv.org/abs/2310.08528)

### Key Implementations

- **Official Reference**: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **Optimized gsplat**: [https://github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- **Awesome List**: [https://github.com/MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)

### Tutorials and Learning Resources

- **LearnOpenCV Tutorial**: [3D Gaussian Splatting - Paper Explained](https://learnopencv.com/3d-gaussian-splatting/)
- **HuggingFace Blog**: [Introduction to 3D Gaussian Splatting](https://huggingface.co/blog/gaussian-splatting)
- **Hands-On Tutorial**: [Radiance Fields and Gaussian Splatting Workshop](https://www.ieeesmc.org/cai-2026/tutorial-6-from-images-to-3d-environments/) (May 2026)

### Recent Surveys and Reviews

- **Recent Advances in 3D Gaussian Splatting** (IEEE, 2026)
  [https://ieeexplore.ieee.org/document/10897713/](https://ieeexplore.ieee.org/document/10897713/)

- **3D Gaussian Splatting Technologies and Extensions: A Review** (2025)
  Comprehensive survey of extensions and applications

- **Gaussian Splatting in February 2026** (Radiance Fields Newsletter)
  [https://radiancefields.substack.com/p/gaussian-splatting-in-february-2026](https://radiancefields.substack.com/p/gaussian-splatting-in-february-2026)

### Community and Updates

- **Radiance Fields Newsletter**: Monthly updates on NeRF, Gaussian Splatting, and related techniques
- **2025 arXiv Papers**: [Daily updated list](https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting)
- **Twitter/X**: Follow #GaussianSplatting and researchers from INRIA, ETH Zurich, MIT CSAIL

---

*This post represents my understanding as of March 2026. The field is evolving rapidly—by the time you read this, new breakthroughs may have emerged. Check the awesome-3D-gaussian-splatting repository for the latest developments.*

---

## Sources

This comprehensive guide was compiled from extensive research including:

- [Gaussian splatting: a complete student guide to 3D capture in 2026 | Medium](https://medium.com/@Jamesroha/gaussian-splatting-a-complete-student-guide-to-3d-capture-in-2026-1195a6265870)
- [3D Gaussian Splatting Tutorial | LearnOpenCV](https://learnopencv.com/3d-gaussian-splatting/)
- [Introduction to 3D Gaussian Splatting | HuggingFace](https://huggingface.co/blog/gaussian-splatting)
- [3D Gaussian Splatting vs NeRF | PyImageSearch](https://pyimagesearch.com/2024/12/09/3d-gaussian-splatting-vs-nerf-the-end-game-of-3d-reconstruction/)
- [Photogrammetry vs. NeRFs vs. Gaussian Splatting | Varjo](https://get.teleport.varjo.com/blog/photogrammetry-vs-nerfs-gaussian-splatting-pros-and-cons)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering | arXiv](https://arxiv.org/abs/2308.04079)
- [Official Implementation | GitHub](https://github.com/graphdeco-inria/gaussian-splatting)
- [Creating stunning real time 3D scenes | INRIA](https://www.inria.fr/en/creating-stunning-real-time-3d-scenes-breakthrough-3d-gaussian-splatting)
- [4D Gaussian Splatting | CVPR 2024](https://github.com/hustvl/4DGaussians)
- [Gaussian Splatting in February 2026 | Radiance Fields](https://radiancefields.substack.com/p/gaussian-splatting-in-february-2026)
- [Recent advances in 3D Gaussian splatting | IEEE](https://ieeexplore.ieee.org/document/10897713/)
- [A Comprehensive Overview of Gaussian Splatting | Towards Data Science](https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362/)
- [VRSplat: Gaussian Splatting for VR | arXiv](https://arxiv.org/abs/2505.10144)
- [gsplat Implementation | GitHub](https://github.com/nerfstudio-project/gsplat)
- [Improving Adaptive Density Control | arXiv](https://arxiv.org/html/2503.14274v1)
- [Spherical Harmonics in Neural Rendering | Medium](https://papers-100-lines.medium.com/explore-how-spherical-harmonics-enhance-neural-radiance-fields-and-3d-gaussian-splatting-by-b33fc755bfc5)

And many other papers, implementations, and industry reports cited throughout the post.
