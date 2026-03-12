# Homework 3: JIT Compilation, Tracing, and GPU Performance Analysis

**Course:** CompSci 790  
**Assignment:** Homework 3  
**Author:** Muhammad Fahad  
**Institution:** University of Wisconsin–Milwaukee  

---

## Overview

This repository contains the complete submission for Homework 3 on JIT compilation, tracing, and GPU performance analysis using **JAX** and **PyTorch**.

The goal of this homework is to study how modern machine learning frameworks optimize computation through tracing and just-in-time compilation, and to evaluate their practical impact on execution time, graph generation, shape specialization, backend behavior, and operator fusion on GPU hardware.

This repository includes:

- `jax_jit_analysis.ipynb`  
  JAX-based experiments on compilation overhead, shape specialization, and operator fusion

- `torch_compile_analysis.ipynb`  
  PyTorch-based experiments on `torch.compile`, backend comparison, debugging graph breaks, and computation graph inspection

- `report.pdf`  
  Final written report summarizing methodology, results, observations, and discussion

- `figures/`  
  Output plots, screenshots, and visual artifacts used in the report

---

## Repository Structure

```text
├── README.md
├── jax_jit_analysis/
│   ├── jax_jit_analysis.ipynb
│   ├── figures/
│   │   ├── jax_part1_execution_time.png
│   │   ├── jax_part2_shape_specialization.png
│   │   └── jax_part3_fusion_timing.png
│   ├── results/
│   │   ├── jax_part1_results.csv
│   │   ├── jax_part2_jaxpr_text.csv
│   │   ├── jax_part2_shape_specialization.csv
│   │   ├── jax_part3_fusion_timing.csv
│   │   ├── jax_part3_hlo.txt
│   │   └── jax_part3_profiler_summary.json
│   └── sample_data/
├── torch_compile_analysis/
│   ├── torch_compile_analysis.ipynb
│   ├── figures/
│   │   └── torch_part1_backend_comparison.png
│   ├── results/
│   │   ├── torch_part1_backend_results.csv
│   │   ├── torch_part2_debugging_results.csv
│   │   ├── torch_part2_dynamo_explain.txt
│   │   ├── torch_part3_dynamo_explain.txt
│   │   └── torch_part3_fx_graph.txt
│   └── sample_data/
└── report.pdf
