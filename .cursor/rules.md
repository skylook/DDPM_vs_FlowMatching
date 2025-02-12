# Project Structure and Guidelines

## Project Overview
This project demonstrates and compares DDPM (Denoising Diffusion Probabilistic Models) and Flow Matching for fitting elliptical distributions. The implementation shows that Flow Matching is more efficient than DDPM in terms of training speed and sampling quality.

## Directory Structure
```
.
├── README.md                 # Project documentation and results
├── train_ddpm.py            # DDPM training and visualization script
├── train_flow_matching.py   # Flow Matching training and visualization script
├── doc/                     # Documentation and references
│   ├── papers/             # Academic papers
│   └── articles/           # Related articles and blog posts
├── ddim_samples/           # DDPM/DDIM sampling results
├── ddim_trajectories/      # DDPM/DDIM sampling trajectories
├── flow_matching_samples/  # Flow Matching sampling results
└── flow_matching_trajectories/ # Flow Matching sampling trajectories
```

## Key Components

### Training Scripts
- `train_ddpm.py`: Implements DDPM training with DDIM sampling
- `train_flow_matching.py`: Implements Flow Matching training and sampling

### Visualization
Both scripts include visualization capabilities:
- Sample generation visualization
- Trajectory visualization for sampling process
- Training progress visualization

## Documentation References

### Academic Papers
1. Flow Matching for Generative Modeling (Lipman et al., 2022)
   - Key paper introducing Flow Matching methodology
   - Location: `doc/Flow Matching for Generative Modeling - arXiv-2210.02747v2/`

2. Flow Straight and Fast (Liu et al., 2022)
   - Research on optimizing flow-based generation
   - Location: `doc/Flow Straight and Fast - Learning to Generate and Transfer Data with Rectified Flow - arXiv-2209.03003v1/`

3. Rectified Flow (2022)
   - A Marginal Preserving Approach to Optimal Transport
   - Location: `doc/Rectified Flow-A Marginal Preserving Approach to Optimal Transport-arXiv-2209.14577v1/`

4. Diffusion Meets Flow Matching
   - Comparative analysis of diffusion and flow matching approaches
   - Location: `doc/Diffusion Meets Flow Matching/`

### Technical Articles
Key reference materials from kexue.fm:
1. Flow Matching Introduction and Analysis
   - URL: https://kexue.fm/archives/9370
   - Key concepts and mathematical foundations

2. Flow Matching Implementation Details
   - URL: https://kexue.fm/archives/9379
   - Practical implementation guidance

3. Advanced Flow Matching Topics
   - URL: https://kexue.fm/archives/9497
   - Advanced concepts and optimizations

4. Diffusion Model Fundamentals
   - URL: https://kexue.fm/archives/9228
   - Background on diffusion models

5. Diffusion Model Implementation
   - URL: https://kexue.fm/archives/9209
   - Practical aspects of diffusion models

## Writing Guidelines for Diffusion-Meets-Flow-Matching.md

### Structure
1. Introduction
   - Background on generative models
   - Motivation for comparing diffusion and flow matching

2. Theoretical Foundation
   - DDPM principles
   - Flow Matching principles
   - Mathematical connections

3. Implementation Comparison
   - Training process differences
   - Sampling strategy differences
   - Efficiency analysis

4. Experimental Results
   - Performance metrics
   - Visual comparisons
   - Training efficiency

5. Conclusions
   - Key findings
   - Future directions

### Content Requirements
- Write in Chinese
- Include mathematical formulas where necessary
- Provide code examples when relevant
- Include visualizations from experiments
- Reference original papers and articles appropriately

## Implementation Guidelines

### Data Generation
- Generate elliptical distribution samples for training
- Implement random sampling for visualization

### Model Training
- DDPM:
  - Use standard diffusion training process
  - Implement DDIM sampling for faster inference
- Flow Matching:
  - Implement velocity field estimation
  - Use optimal transport principles for sampling

### Visualization Requirements
- Generate comparison videos showing:
  - Training progress
  - Sampling trajectories
  - Distribution fitting quality

## Experimental Results
Key findings from experiments:
1. Flow Matching converges faster (around 1000 iterations)
2. Flow Matching produces straighter sampling trajectories
3. Flow Matching achieves more uniform distribution fitting

## Code Style Guidelines
- Use Python 3.7+
- Follow PEP 8 conventions
- Include docstrings for major functions
- Add comments for complex algorithms 