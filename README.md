
# Aircraft Control System Simulation (LQR, LQG, Kalman Filter)

This repository contains MATLAB and Python implementations for designing, simulating, and analyzing control systems for an aircraft’s longitudinal (pitch) and lateral (sideslip, roll) dynamics.

The project demonstrates how Linear Quadratic Regulator (LQR) and Linear Quadratic Gaussian (LQG) controllers improve stability and performance compared to open-loop dynamics. A Kalman Filter is also implemented to show optimal state estimation under measurement noise.

---

## ✈️ Overview

The goal is to design controllers that stabilize the aircraft while balancing performance and control effort.

- Control Methods: LQR, LQG
- Estimation Method: Kalman Filter
- Languages: MATLAB, Python
- Libraries: MATLAB Control Toolbox, NumPy, SciPy, Matplotlib

---

## 📐 System Models

### 1. Longitudinal Dynamics (Pitch Control)
State vector:
x = [w, q, θ] = [Vertical Velocity, Pitch Rate, Pitch Angle]

System matrices:
A = [[-0.3149, 235.8928, 0], [-0.0034, -0.4282, 0], [0, 1, 0]]
B = [[-5.5079], [0.0021], [0]]
C = [[0, 0, 1]]

### 2. Lateral Dynamics (Sideslip & Roll Control)
State vector:
x = [β, p, r, φ] = [Sideslip Angle, Roll Rate, Yaw Rate, Roll Angle]

System matrices:
A = [[-0.0558, -0.9968, 0.0802, 0.0415],
     [0.5980, -0.1150, -0.0318, 0],
     [-3.0500, 0.3880, -0.4650, 0],
     [0, 0.0805, 1, 0]]
B = [[0.0729, 0], [-4.75, 0.00775], [0.1530, 0.1430], [0, 0]]
C = [[1, 0, 0, 0], [0, 0, 0, 1]]

---

## 🕹️ Implemented Methods

- LQR (Linear Quadratic Regulator)
- LQG (Linear Quadratic Gaussian)
- Kalman Filter

---

## 🚀 Running the Code

### Requirements
- MATLAB (R2020a+ with Control System Toolbox)
- Python 3.x with: pip install numpy scipy matplotlib

### Running MATLAB Scripts
1. Open a .m file in MATLAB
2. Click Run
3. The script will plot open-loop vs closed-loop responses

### Running Python Script
python Python_Scripts/kalman_filtering_lateral_py.py

---

## 📁 Repository Structure

MATLAB_Scripts/
- Longitudinal_motion_lqr.m
- Longitudinal_motion_lqg.m
- Lateral_motion_lqr.m
- Lateral_motion_lqg.m
- Kalman_filtering_Longitudinal.m
- Longitudinal_OpenLoop.m
- Lateral_OpenLoop.m

Python_Scripts/
- kalman_filtering_lateral_py.py

Documentation/
- Report.pdf
- Paper.pdf

---

## ✨ Results

- Open-loop: Aircraft dynamics are unstable and oscillatory
- Closed-loop (LQR & LQG): Systems are stabilized with smooth responses
- Kalman Filter: Effectively filters measurement noise

---

## 📚 References

1. Chrif, L., & Kadda, Z. M. (2014). Aircraft Control System Using LQG and LQR Controller with Optimal Estimation-Kalman Filter Design. Procedia Engineering, 80, 245–257.
2. Chevireddi, V. D., & Kambhampati, N. V. S. S. G. Aircraft Control System Using LQG and LQR Controller with Optimal Estimation-Kalman Filter Design.
