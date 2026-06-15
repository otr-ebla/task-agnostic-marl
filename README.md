# Task-Agnostic Cooperative Multi-Robot Learning under Partial Observability

This repository contains the implementation of a decentralized, task-agnostic cooperative control framework for multi-robot fleets operating in dynamic, human-populated indoor environments. The core objective is to train a single cooperative policy capable of executing a family of spatial tasks (such as area coverage, search, patrol, transport, and rendezvous) without requiring per-task retraining.

The underlying problem is formalized as a task-conditioned Decentralized Partially Observable Markov Decision Process (Dec-POMDP) and solved using Multi-Agent Reinforcement Learning (MARL).

---

## Core Framework Architecture

The complete system is designed around Centralized Training with Decentralized Execution (CTDE) and comprises three primary components:

1. **Predictive World Model:** A representation layer that converts local 2D LiDAR scans and odometry into dynamics-aware latent states, encoding short-horizon pedestrian trajectories.
2. **Relational Graph-Transformer Scene Encoder:** A structural network that processes surrounding entities (teammates, humans, and landmarks) as permutation-equivariant tokens, rendering the policy invariant to team size and crowd density.
3. **Task-Conditioned Multi-Agent Policy:** A decentralized controller mapping the latent scene representation and a continuous goal embedding ($g$) to continuous differential-drive velocity commands using MAPPO/HAPPO.

---

## Repository Structure

```text
task-agnostic-marl/
├── README.md                   # Core project documentation
├── .gitignore                  # Build, cache, and logging exclusions
├── docs/                       # Research papers, formulas, and math notes
├── shared-env/                 # Shared Headed Social Force Model (HSFM) pedestrian simulator
└── internal-simple-project/    # Phase 1: Isolated sandbox for algorithmic verification
    ├── requirements.txt        # Minimal baseline dependencies
    ├── config/                 # Hyperparameter configuration files
    └── src/
        └── train_simple.py      # Standalone MAPPO training script (Vector-based state)



