# prompt-unet

A repository for development and evaluation of Prompt-UNet models.

## Evaluation & Benchmarks

This project includes benchmarking of external models such as **nnInteractive**.

For instructions on how to set up and use `nnInteractive` for benchmarking, see the [nnInteractive Setup Guide](file:///evaluation/benchmark_models/NNINTERACTIVE_SETUP.md).

## Project Structure
- `data/`: Dataset storage and processing scripts.
- `deployment/`: Model export and deployment assets.
- `evaluation/`: scripts and models for benchmarking.
  - `benchmark_models/`: Setup and configurations for external models.
- `training/`: Training routines for Prompt-UNet.
- `utils/`: Common utilities and augmentation testing.