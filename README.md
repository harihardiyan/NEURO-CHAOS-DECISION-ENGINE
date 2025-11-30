# 🧠 NEURO-CHAOS Decision Engine: A Bistable Dynamical System for Autonomous Regulation

## Project Overview

This repository contains the source code for the NEURO-CHAOS Decision Engine, an advanced, autonomous control system designed to detect and respond to incipient chaotic states in critical operational environments.

The architecture integrates two primary subsystems:
1.  **Chaos Proof (V31B):** A real-time non-linear diagnostic tool that calculates the **Largest Lyapunov Exponent (LLE)** to quantify the severity of chaos from time-series data.
2.  **Neuro-Dyn (V33b):** A **Bistable Dynamical System** (a system of Ordinary Differential Equations, ODEs) that receives the LLE signal and determines the final operational state (Safe State $C_2$ or Critical Action $C_1$).

The system is calibrated with an **LLE Signal Filter Threshold ($S_{min}=0.0600$)** to prevent hyper-paranoia caused by sensor anomalies (e.g., LLE = 0.0576 under stable conditions).

---

## 📐 Governing Equations (Academic Rigor)

The system's core logic is governed by two sets of equations: the LLE estimation and the Bistable ODEs.

### 1. Chaos Proof (V31B) - Largest Lyapunov Exponent (LLE)

The LLE ($\lambda$) is calculated by tracking the average exponential separation of nearby trajectories in the reconstructed phase space.

$$\lambda = \frac{1}{\tau \cdot (\text{Iteration})} \sum_{i=1}^{\text{Iteration}} \ln \left( \frac{d'(i)}{d(i)} \right)$$

* $d(i)$: Initial distance between nearest neighbors.
* $d'(i)$: Distance after one evolution time $\tau$.
* $\tau$: Time delay used for phase space reconstruction (e.g., $\tau=2$).

### 2. Neuro-Dyn - Bistable ODE System

The dynamics of the two competitive states, Critical ($C_1$) and Stable ($C_2$), are modeled by the following coupled ODEs:

$$\begin{aligned}
\frac{dC_1}{dt} &= \frac{\alpha_1}{1 + (C_2 / K_1)^{\eta_1}} + \beta \cdot S_{\text{filtered}} - \gamma_1 C_1 \\
\frac{dC_2}{dt} &= \frac{\alpha_2}{1 + (C_1 / K_2)^{\eta_2}} - \gamma_2 C_2
\end{aligned}$$

Where:
* $S_{\text{filtered}}$: The chaos signal input, derived from the LLE:
$$\mathbf{S_{\text{filtered}} = \max(0.0, \text{LLE}_{\text{raw}} - S_{\text{min}})}$$
* $\beta$: Coupling rate (set to **120.0** for high gain/gradated response).
* $C_0$: Initial condition, set to **$[6.0, 4.0]$** (Biased towards $C_1$ to ensure LLE overcomes basal stability).
* $\alpha, K, \gamma, \eta$: System parameters defining expression rates, repression thresholds, decay rates, and cooperativity (all locked to stable values, e.g., $K_1=K_2=5.0$, $\alpha_i=10.0$).

---

## 🎯 Final Rigor Calibration

| Condition | LLE Raw | LLE Filtered (S) | Final $C_1$ | Final $C_2$ | Action Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stable (RR=1.00)** | $0.0576$ | $\mathbf{0.0000}$ | $\approx 0.25$ | $\approx 10.00$ | **CRUISE MODE (Stable)** |
| **Bounded Chaos (RR=0.50)** | $0.0614$ | $\mathbf{0.0014}$ | $\approx 10.16$ | $\approx 0.20$ | **HARD WARNING** |
| **Runaway Chaos (RR=0.00)** | $0.0872$ | $\mathbf{0.0272}$ | $\approx 13.26$ | $\approx 0.16$ | **IMMEDIATE SCRAM** |

## 👤 Author Information

* **Author:** Hari Hardiyan
* **Email:** lorozloraz@gmail.com

---

## ⚙️ Running the Pipeline

Dependencies: `numpy` and `scipy`.

```bash
# Clone the repository
git clone [https://github.com/YourUsername/NEURO-CHAOS-DECISION-ENGINE.git](https://github.com/YourUsername/NEURO-CHAOS-DECISION-ENGINE.git)
cd NEURO-CHAOS-DECISION-ENGINE
```

# Install dependencies
pip install numpy scipy

# Run the final integrated benchmark

python neuro_chaos_pipeline.py
