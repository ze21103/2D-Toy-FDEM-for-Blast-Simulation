# 2D Toy FDEM for Blast Simulation 🧨

**A minimalistic Python-based implementation of the Finite-Discrete Element Method (FDEM) for blast and fracture analysis.**

This repository provides a suite of lightweight 2D FDEM tutorial models. By stripping away complex industrial-grade architectures, these scripts focus on the core physical mechanisms: **continuum-to-discrete transition, stress wave propagation, contact mechanics, and structural fragmentation**. Specifically tailored for **multi-borehole internal blasting in concrete**, these models demonstrate the full dynamic evolution from initial detonation to crack branching and final fragment interaction.

---

## 📂 Project Structure & Learning Roadmap

The project is divided into four standalone Python scripts, designed to be studied in sequential order to master the core logic of FDEM:

### `01_mini_fdem_blast.py` | Fundamental Fracture Mechanics
* **Mesh**: Structured grid (31x31 nodes).
* **Core Logic**: Implementation of the Spring-Mass system, Hooke's law, and the Maximum Tensile Strain failure criterion.
* **Scenario**: A single-point blast at the center of a concrete block to observe the initial fracture network.

### `02_unstructured_fdem_3holes.py` | Unstructured Mesh & Wave Interaction
* **Mesh**: Unstructured Delaunay triangular mesh to eliminate mesh-alignment bias.
* **Core Logic**: Improved isotropic representation of material failure.
* **Scenario**: **Three-borehole blast configuration**. It visualizes complex crack branching caused by the interaction of multiple stress waves.

### `03_fdem_contact_tutorial.py` | Contact Mechanics 101
* **Core Logic**: Focuses exclusively on the **Penalty Method** for contact detection and force calculation.
* **Scenario**: Simulates the high-speed collision of two independent blocks (Block A and Block B) to demonstrate overlap calculation and rebound effects.

### `04_coupled_fdem_3holes.py` | Ultimate Coupled Model
* **Core Logic**: Integrates dynamic fracture with a high-performance **vectorized global contact search**. It uses an upper triangular distance matrix to avoid redundant computations.
* **Scenario**: A full simulation of three-borehole fragmentation. The material not only fractures but also accounts for the collision and repulsion of flying debris, providing high physical fidelity for structural collapse.

---

## 🛠️ Installation & Usage

### 1. Prerequisites
The models rely only on standard scientific computing libraries:

```bash
pip install numpy scipy matplotlib
```

### 2. Running the Simulations
Execute the scripts directly via the Python interpreter. For example, to run the ultimate coupled model:

```bash
python 04_coupled_fdem_3holes.py
```

### 3. Outputs
* **Terminal**: Real-time tracking of calculation steps and the number of macro-cracks (broken springs).
* **Visualization**: High-resolution comparison plots (Initial vs. Post-Blast) are rendered via `matplotlib` and automatically saved to the local directory.
