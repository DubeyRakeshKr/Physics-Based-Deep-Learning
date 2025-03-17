# Physics-Based Deep Learning

## About The Project

This repository focuses on **Physics-Based Deep Learning (PBDL)**, collecting resources and papers on methods combining physical modeling with deep learning techniques. PBDL represents an active and rapidly growing research field.

### Major Contributor
This repository was significantly contributed to by **Rakesh Dubey**, who helped organize and compile research resources in the field of Physics-Based Deep Learning.

## Overview

If you're interested in a comprehensive overview, check our digital PBDL book: [Physics-Based Deep Learning](https://www.physicsbaseddeeplearning.org/) or as [PDF](https://arxiv.org/pdf/2109.05237.pdf)

Within this area, we distinguish different physics-based approaches:

- **Forward** simulations (predicting state or temporal evolution)
- **Inverse** problems (obtaining parametrization from observations)

The integration between learning and physics can be categorized as:

- **Data-driven**: Data produced by a physical system with no further interaction
- **Loss-terms**: Physical dynamics encoded in the loss function via differentiable operations
- **Interleaved**: Full physical simulation combined with neural network output (requires a differentiable simulator)

This repository primarily collects links to work from the I15 lab at TUM and other research groups, with emphasis on fluid flow (Navier-Stokes related problems).

## Research Papers

### I15 Physics-based Deep Learning Links

Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks , 
Project: <https://github.com/tum-pbs/dgn4cfd>

Flow Matching for Posterior Inference with Simulator Feedback , 
PDF: <https://arxiv.org/pdf/2410.22573>

APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs , 
PDF: <http://arxiv.org/pdf/2411.00180>

Deep learning-based predictive modelling of transonic flow over an aerofoil , 
PDF: <https://arxiv.org/pdf/2403.17131>

ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks , 
Project: <https://tum-pbs.github.io/ConFIG/>

The Unreasonable Effectiveness of Solving Inverse Problems with Neural Networks ,
PDF: <http://arxiv.org/pdf/2408.08119> 

Phiflow: Differentiable Simulations for PyTorch, TensorFlow and Jax ,
PDF: <https://openreview.net/pdf/36503358a4f388f00d587a0257c13ba2a4656098.pdf>

How Temporal Unrolling Supports Neural Physics Simulators , 
PDF: <https://arxiv.org/pdf/2402.12971>

Stabilizing Backpropagation Through Time to Learn Complex Physics , 
PDF: <https://openreview.net/forum?id=bozbTTWcaw>

Symmetric Basis Convolutions for Learning Lagrangian Fluid Mechanics , 
PDF: <https://openreview.net/forum?id=HKgRwNhI9R> 

Uncertainty-aware Surrogate Models for Airfoil Flow Simulations with Denoising Diffusion Probabilistic Models , 
Project: <https://github.com/tum-pbs/Diffusion-based-Flow-Prediction>

Turbulent Flow Simulation using Autoregressive Conditional Diffusion Models , 
Project: <https://github.com/tum-pbs/autoreg-pde-diffusion>

Physics-Preserving AI-Accelerated Simulations of Plasma Turbulence , 
PDF: <https://arxiv.org/pdf/2309.16400>>

Unsteady Cylinder Wakes from Arbitrary Bodies with Differentiable Physics-Assisted Neural Network , 
Project: <https://github.com/tum-pbs/DiffPhys-CylinderWakeFlow>

Score Matching via Differentiable Physics , 
Project: <https://github.com/tum-pbs/SMDP> 

Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics , 
Project: <https://github.com/tum-pbs/DMCF>

Learned Turbulence Modelling with Differentiable Fluid Solvers , 
Project: <https://github.com/tum-pbs/differentiable-piso> 

Half-Inverse Gradients for Physical Deep Learning , 
Project: <https://github.com/tum-pbs/half-inverse-gradients> 

Reviving Autoencoder Pretraining (Previously: Data-driven Regularization via Racecar Training for Generalizing Neural Networks), 
Project: <https://github.com/tum-pbs/racecar>

Realistic galaxy images and improved robustness in machine learning tasks from generative modelling , 
PDF: <https://arxiv.org/pdf/2203.11956>

Hybrid Neural Network PDE Solvers for Reacting Flows , 
Project: <https://github.com/tum-pbs/Hybrid-Solver-for-Reactive-Flows> 

Scale-invariant Learning by Physics Inversion (formerly "Physical Gradients") ,
Project: <https://github.com/tum-pbs/SIP>

High-accuracy transonic RANS Flow Predictions with Deep Neural Networks ,
Project: <https://github.com/tum-pbs/coord-trans-encoding> 

Learning Meaningful Controls for Fluids ,
Project: <https://rachelcmy.github.io/den2vel/>

Global Transport for Fluid Reconstruction with Learned Self-Supervision ,
Project: <https://ge.in.tum.de/publications/2021-franz-globtrans>

Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers , 
Project: <https://github.com/tum-pbs/Solver-in-the-Loop>

Numerical investigation of minimum drag profiles in laminar flow using deep learning surrogates ,
PDF: <https://arxiv.org/pdf/2009.14339>

Purely data-driven medium-range weather forecasting achieves comparable skill to physical models at similar resolution , 
PDF: <https://arxiv.org/pdf/2008.08626>

Latent Space Subdivision: Stable and Controllable Time Predictions for Fluid Flow , 
Project: <https://ge.in.tum.de/publications/2020-lssubdiv-wiewel>

WeatherBench: A benchmark dataset for data-driven weather forecasting , 
Project: <https://github.com/pangeo-data/WeatherBench>

Learning Similarity Metrics for Numerical Simulations (LSiM) ,
Project: <https://ge.in.tum.de/publications/2020-lsim-kohl>

Learning to Control PDEs with Differentiable Physics , 
Project: <https://ge.in.tum.de/publications/2020-iclr-holl>

Lagrangian Fluid Simulation with Continuous Convolutions , 
PDF: <https://openreview.net/forum?id=B1lDoJSYDH>

Tranquil-Clouds: Neural Networks for Learning Temporally Coherent Features in Point Clouds , 
Project: <https://ge.in.tum.de/publications/2020-iclr-prantl/>

ScalarFlow: A Large-Scale Volumetric Data Set of Real-world Scalar Transport Flows for Computer Animation and Machine Learning , 
Project: <https://ge.in.tum.de/publications/2019-tog-eckert/>

tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow , 
Project: <https://ge.in.tum.de/publications/tempogan/>

Deep Fluids: A Generative Network for Parameterized Fluid Simulations , 
Project: <http://www.byungsoo.me/project/deep-fluids/>

Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow , 
Project: <https://ge.in.tum.de/publications/latent-space-physics/>

A Multi-Pass GAN for Fluid Flow Super-Resolution , 
PDF: <https://ge.in.tum.de/publications/2019-multi-pass-gan/>

A Study of Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations , 
Project: <https://github.com/thunil/Deep-Flow-Prediction>

Data-Driven Synthesis of Smoke Flows with CNN-based Feature Descriptors , 
Project: <http://ge.in.tum.de/publications/2017-sig-chu/>

Liquid Splash Modeling with Neural Networks , 
Project: <https://ge.in.tum.de/publications/2018-mlflip-um/>

Generating Liquid Simulations with Deformation-aware Neural Networks , 
Project: <https://ge.in.tum.de/publications/2017-prantl-defonn/>

### Additional Links for Fluids

Discretize first, filter next: Learning divergence-consistent closure models for large-eddy simulation , 
PDF: <https://doi.org/10.1016/j.jcp.2024.113577>

Data-Efficient Inference of Neural Fluid Fields via SciML Foundation Model , 
PDF: <https://arxiv.org/abs/2412.13897>

DeepLag: Discovering Deep Lagrangian Dynamics for Intuitive Fluid Prediction ,
PDF: <https://arxiv.org/pdf/2402.02425>

Inferring Hybrid Neural Fluid Fields from Videos , 
PDF: <https://openreview.net/pdf?id=kRdaTkaBwC> 

LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite , 
Project: <https://github.com/tumaer/lagrangebench>

CFDBench: A Comprehensive Benchmark for Machine Learning Methods in Fluid Dynamics , 
PDF: <https://arxiv.org/pdf/2310.05963.pdf>

Physics-guided training of GAN to improve accuracy in airfoil design synthesis , 
PDF: <https://arxiv.org/pdf/2308.10038>

A probabilistic, data-driven closure model for RANS simulations with aleatoric, model uncertainty , 
PDF: <https://arxiv.org/pdf/2307.02432>

Differentiable Turbulence , 
PDF: <https://arxiv.org/pdf/2307.03683>

Super-resolving sparse observations in partial differential equations: A physics-constrained convolutional neural network approach ,
PDF: <https://arxiv.org/pdf/2306.10990>

Reduced-order modeling of fluid flows with transformers ,
PDF: <https://pubs.aip.org/aip/pof/article/35/5/057126/2891586>

Machine learning enhanced real-time aerodynamic forces prediction based on sparse pressure sensor inputs , 
PDF: <https://arxiv.org/pdf/2305.09199>

Reconstructing Turbulent Flows Using Physics-Aware Spatio-Temporal Dynamics and Test-Time Refinement , 
PDF: <https://arxiv.org/pdf/2304.12130>

Inferring Fluid Dynamics via Inverse Rendering , 
PDF: <https://arxiv.org/pdf/2304.04446>

Multi-scale rotation-equivariant graph neural networks for unsteady Eulerian fluid dynamics , 
WWW: <https://aip.scitation.org/doi/10.1063/5.0097679>

[Additional papers listed in original README...]



## Contact

Physics-based deep learning is a very dynamic field. Please let us know if we've overlooked papers that you think should be included by sending a mail to _i15ge at cs.tum.de_, and feel free to check out our homepage at <https://ge.in.tum.de/>.

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/tum-pbs/physics-based-deep-learning.svg?style=flat-square
[contributors-url]: https://github.com/tum-pbs/physics-based-deep-learning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/tum-pbs/physics-based-deep-learning.svg?style=flat-square
[forks-url]: https://github.com/tum-pbs/physics-based-deep-learning/network/members
[stars-shield]: https://img.shields.io/github/stars/tum-pbs/physics-based-deep-learning.svg?style=flat-square
[stars-url]: https://github.com/tum-pbs/physics-based-deep-learning/stargazers
[issues-shield]: https://img.shields.io/github/issues/tum-pbs/physics-based-deep-learning.svg?style=flat-square
[issues-url]: https://github.com/tum-pbs/physics-based-deep-learning/issues
[license-shield]: https://img.shields.io/github/license/tum-pbs/physics-based-deep-learning.svg?style=flat-square
[license-url]: https://github.com/tum-pbs/physics-based-deep-learning/blob/master/LICENSE
