# prompt-unet

A repository for development and evaluation of the Prompt U-Net.

## Evaluation & Benchmarks

This project includes benchmarking of external the models **nnInteractive** and **UniverSeg**.

For instructions on how to set up and use them for benchmarking, see the under evaluation/benchmark_models.

## Project Structure
- `data/`: Dataset storage and processing scripts.
- `deployment/`: Model export and deployment assets.
- `evaluation/`: scripts and models for benchmarking.
  - `benchmark_models/`: Setup and configurations for external models.
- `training/`: Training routines for Prompt-UNet.
- `utils/`: Common utilities and augmentation testing.
