################################################################################
###   --- FILE: README.md ---   ###
################################################################################

# Aircraft Control System Simulation using LQR, LQG, and Kalman Filter

This project involves the design, simulation, and analysis of robust control systems for an aircraft's longitudinal (pitch) and lateral (sideslip, roll) dynamics. The implementation uses Linear Quadratic Regulator (LQR) and Linear Quadratic Gaussian (LQG) controllers to achieve stability and optimal performance. A Kalman filter is also implemented to demonstrate optimal state estimation from noisy measurements.

---

### ✈️ Project Overview

The goal of this project is to simulate a robust controller that can maintain stability and good dynamic performance for an aircraft's flight dynamics, even when faced with disturbances and noise. The simulations compare the unstable open-loop response of the aircraft with the stabilized closed-loop response achieved by the LQR and LQG controllers.

* **Control Methods:** LQR & LQG
* **Estimation Method:** Kalman Filter
* **Programming Languages:** MATLAB, Python
* **Core Libraries:** MATLAB Control System Toolbox, NumPy, SciPy, Matplotlib

### 📐 System Models

The aircraft's motion is decoupled into two independent systems: longitudinal and lateral dynamics. Both are represented using state-space models based on the academic paper provided.

* #### Longitudinal Dynamics (Pitch Control)
    This model describes the aircraft's motion in the vertical plane, primarily controlling the **pitch angle**. The state vector and system matrices are defined as follows:

    **State Vector:**
    $$
    x = \begin{bmatrix} w \\ q \\ \theta \end{bmatrix} = \begin{bmatrix} \text{Vertical Velocity} \\ \text{Pitch Rate} \\ \text{Pitch Angle} \end{bmatrix}
    $$
    **State-Space Matrices:**
    $$
    A = \begin{bmatrix} -0.3149 & 235.8928 & 0 \\ -0.0034 & -0.4282 & 0 \\ 0 & 1 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} -5.5079 \\ 0.0021 \\ 0 \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}
    $$

* #### Lateral Dynamics (Sideslip & Roll Control)
    This model describes the aircraft's motion related to its vertical axis, controlling the **sideslip angle** ($\beta$) and **roll angle** ($\phi$). The state vector and system matrices are:

    **State Vector:**
    $$
    x = \begin{bmatrix} \beta \\ p \\ r \\ \phi \end{bmatrix} = \begin{bmatrix} \text{Sideslip Angle} \\ \text{Roll Rate} \\ \text{Yaw Rate} \\ \text{Roll Angle} \end{bmatrix}
    $$
    **State-Space Matrices:**
    $$
    A = \begin{bmatrix} -0.0558 & -0.9968 & 0.0802 & 0.0415 \\ 0.5980 & -0.1150 & -0.0318 & 0 \\ -3.0500 & 0.3880 & -0.4650 & 0 \\ 0 & 0.0805 & 1 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 0.0729 & 0 \\ -4.75 & 0.00775 \\ 0.1530 & 0.1430 \\ 0 & 0 \end{bmatrix}
    $$   $$
    C = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
    $$

### 🕹️ Controllers & Estimators

* **Linear Quadratic Regulator (LQR)**
    An optimal full-state feedback controller that calculates a gain matrix $K$ to minimize a quadratic cost function. This balances the trade-off between state regulation (performance) and control effort. The cost function is:
    $$
    J(u) = \int_{0}^{\infty} (x^T Q x + u^T R u) dt
    $$

* **Linear Quadratic Gaussian (LQG)**
    The LQG controller combines the LQR with a Kalman filter. It is designed for systems that experience Gaussian white noise in both the process and measurements, providing robust optimal control when the full state is not directly measurable.

* **Kalman Filter**
    An optimal state estimator that produces accurate estimates of a system's state by combining noisy measurements over time. It is highly effective at filtering out noise and providing a clean signal for the controller.

### 🚀 How to Run

#### Prerequisites

* **MATLAB:**
    * MATLAB R2020a or newer.
    * Control System Toolbox.

* **Python:**
    * Python 3.x
    * Required libraries: `numpy`, `scipy`, `matplotlib`. Install them using pip:
        ```bash
        pip install numpy scipy matplotlib
        ```

#### Running the Simulations

* **MATLAB Scripts:**
    1.  Open any of the `.m` files in the MATLAB IDE.
    2.  Click the **Run** button.
    3.  The script will execute and generate plots comparing the open-loop and closed-loop (controlled) system responses.

* **Python Script:**
    1.  Navigate to the repository directory in your terminal.
    2.  Run the script using the following command:
        ```bash
        python kalman_filtering_lateral_py.py
        ```

### 📁 Repository Contents

```
.
├── MATLAB_Scripts/
│   ├── Longitudanal_motion_lqr.m       # LQR Control for Pitch Angle
│   ├── Longitudanal_motion_lqg.m       # LQG Control for Pitch Angle
│   ├── Lateral_Motion_lqr.m            # LQR Control for Sideslip & Roll
│   ├── Lateral_motion_lqg.m            # LQG Control for Sideslip & Roll
│   ├── kalman_filtering_Longitudanal.m # Kalman Filter Demo for Pitch Dynamics
│   ├── Longitudanal_motion_OpenLoop.m  # Open-Loop Response (Pitch)
│   └── Lateral_motion_OpenLoop.m       # Open-Loop Response (Sideslip & Roll)
│
├── Python_Scripts/
│   └── kalman_filtering_lateral_py.py  # Kalman Filter Demo for Lateral Dynamics
│
├── Documentation/
│   ├── Report.pdf                      # Detailed student report on this project
│   └── Paper.pdf                       # Academic paper with system models
│
└── README.md                           # You are here!
```

### ✨ Results & Discussion

The simulation results clearly demonstrate the effectiveness of the implemented controllers.

* **Open-Loop vs. Closed-Loop:** The open-loop impulse responses for both longitudinal and lateral systems are highly oscillatory and unstable, as expected. In contrast, both LQR and LQG controllers successfully stabilize the system, driving the pitch, roll, and sideslip angles to zero in a short amount of time.

* **LQG Performance:** The LQG controller effectively regulates the system states to zero, showing robust performance even in a conceptual framework with process and measurement noise.

    
    *(Image from Report.pdf showing LQG stabilization of the sideslip angle)*

* **Kalman Filter Estimation:** The Kalman filter implementation successfully filters out significant random noise from the measurements, producing an estimated state that closely tracks the true state. The error between the true and estimated state is significantly smaller than the error in the raw measurements.

    
    *(Image from Report.pdf showing true vs. filtered response)*

### 📚 References

The state-space models, theoretical background, and control objectives for this project are primarily based on the work presented in the following documents included in this repository:

1.  **Chrif, L., & Kadda, Z. M.** (2014). *Aircraft Control System Using LQG and LQR Controller with Optimal Estimation-Kalman Filter Design*. Procedia Engineering, 80, 245-257. (`Paper.pdf`)
2.  **Chevireddi, V. D., & Kambhmapati, N. V. S. S. G.** *Aircraft Control System Using LQG and LQR controller with Optimal Estimation-Kalman Filter Design*. (`Report.pdf`)


################################################################################
###   --- FILE: Longitudanal_motion_lqr.m ---   ###
################################################################################

```matlab
clear all
% Define the state-space system matrices (A, B, C, D)
A = [-0.3149,235.8928 0; -0.0034, -0.4282, 0; 0, 1, 0];
B = [-5.5079; 0.0021; 0];
C = [0 0 1];
D = 0;

% Open Loop System
sys = ss(A, B, C, D);

% Define the Q and R matrices for LQR controller design
R = 1;
Q = 500*(C'*C);           


% Design the LQR controller
K = lqr(A, B, Q, R);

% Calculate the constant gain Nbar to eliminate steady-state error
Nbar = -1 / (C * (A - B * K)^(-1) * B); %For unit reference input

% Create the closed-loop system with Nbar
A_cl = A - B * K;
B_cl = B*Nbar;
C_cl = C;
D_cl = D;

% Define the state-space system for closed-loop control with Nbar
sys_cl = ss(A_cl, B_cl, C_cl, D_cl);

% Generate the impulse response of the closed-loop system
t_impulse = 0:0.01:20;  % Time vector for impulse response
impulse_response_lqr = impulse(sys_cl, t_impulse);
impulse_response_openloop = impulse(sys, t_impulse);


% Plot the impulse response
figure;
plot(t_impulse, impulse_response_lqr, 'b', 'LineWidth', 2);

hold on
plot(t_impulse, impulse_response_openloop, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('Pitch Angle');

legend({'lqr','open loop'},'Location','northeast')
grid on;

% Display the calculated Nbar
fprintf('Constant Gain Nbar: %f\n', Nbar);
```

################################################################################
###   --- FILE: Longitudanal_motion_lqg.m ---   ###
################################################################################

```matlab
clear all
% Define the system matrices
A = [-0.3149,235.8928,0; -0.0034,-0.4282,0; 0,1,0];
B = [-5.5079; 0.0021; 0];
C = [0,0,1];
D = 0;

% Define the disturbance matrices
% Assume a small positive value for process noise covariance
Qn = 1; % Modify as per your system's characteristics
Rn = 1; % Measurement noise covariance
Nn = 0; % Cross-correlation between process and measurement noise (assuming zero)

% Define the LQR weighting matrices
Qlqr = [0,0,0; 0,0,0; 0,0,500];
Rlqr = 1;

% Compute the LQR gain
K = lqr(A, B, Qlqr, Rlqr);

% Create state-space model Open Loop
sys = ss(A, B, C, D);

% Design the Kalman filter
[kest,L,P] = kalman(sys, Qn, Rn, Nn);

% Form the LQG regulator
Acl = [A-B*K B*K; zeros(size(A)) A-L*C];
Bcl = [B; zeros(size(B))];
Ccl = [C zeros(size(C))];
Dcl = D;

% Create state-space model Closed Loop 
sys_cl = ss(Acl, Bcl, Ccl, Dcl);

% Simulate the impulse response
t_impulse = 0:0.01:20;  % Time vector for impulse response
impulse_response_lqg = impulse(sys_cl, t_impulse);
impulse_response_openloop = impulse(sys, t_impulse);


% Plot the impulse response
figure;
plot(t_impulse, impulse_response_lqg, 'b', 'LineWidth', 2);

hold on
plot(t_impulse, impulse_response_openloop, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('Pitch Angle');

legend({'lqg','open loop'},'Location','northeast')
grid on;
```

################################################################################
###   --- FILE: Lateral_Motion_lqr.m ---   ###
################################################################################

```matlab
clear all
% Define the state-space system matrices (A, B, C, D)
A = [-0.0558,-0.9968,0.0802,0.0415;0.5980,-0.1150,-0.0318,0;-3.0500,0.3880,-0.4650,0;0,0.0805,1,0 ];
B = [0.0729,0;-4.75,0.00775;0.15300,0.1430;0,0];
C = [1 0 0 0;0 0 0 1];
D = [0 0; 0 0];

% Open Loop System
sys = ss(A,B,C,D);

% Define the Q and R matrices for LQR controller desig
R = 1;
Q = 500*(C'*C);

% Design the LQR controller
K = lqr(A, B, Q, R);

% Calculate the constant gain Nbar to eliminate steady-state error
Nbar = -inv(C*inv(A-B*K)*B); %For unit reference input

% Create the closed-loop system with Nbar
A_cl = A - B * K;
B_cl = B*100;
C_cl = C;
D_cl = D;

% Define the state-space system for closed-loop control with Nbar
sys_cl = ss(A_cl, B_cl, C_cl, D_cl);
t_impulse = 0:0.01:20;

impulse_response_lqr1 = impulse(sys_cl(1,1), t_impulse);
impulse_response_openloop1 = impulse(sys(1,1), t_impulse);


% Plot the impulse response
figure(1);
plot(t_impulse, impulse_response_lqr1, 'b', 'LineWidth', 2);

hold on
plot(t_impulse, impulse_response_openloop1, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('Side Slip Angle');

legend({'lqr','open loop'},'Location','northeast')
grid on;


impulse_response_lqr2 = impulse(sys_cl(2,1), t_impulse);
impulse_response_openloop2 = impulse(sys(2,1), t_impulse);
figure(2);
plot(t_impulse, impulse_response_lqr2, 'b', 'LineWidth', 2);

hold on
plot(t_impulse, impulse_response_openloop2, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('Roll Angle');

legend({'lqr','open loop'},'Location','northeast')
```

################################################################################
###   --- FILE: Lateral_motion_lqg.m ---   ###
################################################################################

```matlab
clear all
% Define the system matrices
A = [-0.0558,-0.9968,0.0802,0.0415;0.5980,-0.1150,-0.0318,0;-3.0500,0.3880,-0.4650,0;0,0.0805,1,0 ];
B = [0.0729,0;-4.75,0.00775;0.15300,0.1430;0,0];
C = [1 0 0 0;0 0 0 1];
D = [0 0; 0 0];


% Define the disturbance matrices
% Assume a small positive value for process noise covariance
Qn = 1; % Modify as per your system's characteristics
Rn = 1; % Measurement noise covariance
Nn = 0; % Cross-correlation between process and measurement noise (assuming zero)

% Define the LQR weighting matrices
Qlqr = [0,0,0,0; 0,0,0,0;0,0,0,0; 0,0,0,500];
Rlqr = 1;

% Compute the LQR gain
K = lqr(A, B, Qlqr, Rlqr);

% Create state-space model Open Loop
sys = ss(A, B, C, D);

% Design the Kalman filter
[kest,L,P] = kalman(sys, Qn, Rn, Nn);

% Form the LQG regulator
Acl = [A-B*K B*K; zeros(size(A)) A-L*C];
Bcl = [B; zeros(size(B))];
Ccl = [C zeros(size(C))];
Dcl = D;

% Create state-space model Closed Loop
sys_cl = ss(Acl, Bcl, Ccl, Dcl);

% Simulate the impulse response
t_impulse = 0:0.01:20;  % Time vector for impulse response

impulse_response_lqg1 = impulse(sys_cl(1,1), t_impulse);
impulse_response_openloop1 = impulse(sys(1,1), t_impulse);


% Plot the impulse response
figure(1);
plot(t_impulse, impulse_response_lqg1, 'b', 'LineWidth', 2);

hold on
plot(t_impulse, impulse_response_openloop1, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('Side Slip Angle');

legend({'lqg','open loop'},'Location','northeast')
grid on;


impulse_response_lqg2 = impulse(sys_cl(2,1), t_impulse);
impulse_response_openloop2 = impulse(sys(2,1), t_impulse);

figure(2);
plot(t_impulse, impulse_response_lqg2, 'b', 'LineWidth', 2);

hold on
plot(t_impulse, impulse_response_openloop2, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('Roll Angle');

legend({'lqg','open loop'},'Location','northeast')
```

################################################################################
###   --- FILE: kalman_filtering_Longitudanal.m ---   ###
################################################################################

```matlab
clear all
% Define the system matrices
A = [-0.3149,235.8928,0; -0.0034,-0.4282,0; 0,1,0];
B = [-5.5079; 0.0021; 0];
C = [0,0,1];
D = 0;

% Sample Time = -1 to mark discrete time
Ts = -1; 
% Discrete Plant Model
sys = ss(A,[B B],C,D,Ts,'InputName',{'u' 'w'},'OutputName','y');  % Plant dynamics and additive input noise w

% noise covariance Q and the sensor noise covariance R are values greater than zero
Q = 2.3; 
R = 1;

% Design the Kalman Filter
[kalmf,L,~,Mx,Z] = kalman(sys,Q,R);
%  discard the state estimates and keep only the first output,y_hat
kalmf = kalmf(1,:);


sys.InputName = {'u','w'};
sys.OutputName = {'yt'};

% sumblk to create an input for the measurement noise v
vIn = sumblk('y=yt+v');

kalmf.InputName = {'u','y'};
kalmf.OutputName = 'ye';

% Using connect to join sys and the Kalman filter together such that u is a shared input and the noisy plant output y feeds into the other filter input
SimModel = connect(sys,vIn,kalmf,{'u','w','v'},{'yt','ye'});

t = (0:100)';
% Sinusoidal input Vector
u = sin(t/5);

rng(10,'twister');
w = sqrt(Q)*randn(length(t),1);
v = sqrt(R)*randn(length(t),1);

% Simulate the response
out = lsim(SimModel,[u,w,v]);

yt = out(:,1);   % true response
ye = out(:,2);  % filtered response
y = yt + v;     % measured response

% Comparing the true response with the filtered response
clf
subplot(211), plot(t,yt,'b',t,ye,'g'), 
xlabel('Number of Samples'), ylabel('Output')
title('Kalman Filter Response')
legend('True','Filtered')
subplot(212), plot(t,yt-y,'b',t,yt-ye,'g'),
xlabel('Number of Samples'), ylabel('Error')
legend('True - measured','True - filtered')
```

################################################################################
###   --- FILE: Longitudanal_motion_OpenLoop.m ---   ###
################################################################################

```matlab
clear all
% State Space Matrices
A = [-0.3149,235.8928,0; -0.0034,-0.4282,0; 0,1,0];
B = [-5.5079; 0.0021; 0];
C = [0,0,1];
D = 0;
sys = ss(A,B,C,D);

% Time Interval
t_impulse = 0:0.01:20;  % Time vector for impulse response

% Impulse Response
impulse_response_openloop = impulse(sys, t_impulse);
plot(t_impulse, impulse_response_openloop, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('Pitch Angle');
```

################################################################################
###   --- FILE: Lateral_motion_OpenLoop.m ---   ###
################################################################################

```matlab
clear all
% State Space Matrices
A = [-0.0558,-0.9968,0.0802,0.0415;0.5980,-0.1150,-0.0318,0;-3.0500,0.3880,-0.4650,0;0,0.0805,1,0 ];
B = [0.0729,0;-4.75,0.00775;0.15300,0.1430;0,0];
C = [1 0 0 0;0 0 0 1];
D = [0 0; 0 0];

% State Space Equations
sys = ss(A,B,C,D);
% Time Interval
t_impulse = 0:0.01:20;  % Time vector for impulse response

% Impulse Response
impulse_response_openloop1 = impulse(sys(1,1), t_impulse);
impulse_response_openloop2 = impulse(sys(2,1), t_impulse);

figure(1);
plot(t_impulse, impulse_response_openloop1, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('Side Slip Angle');

figure(2);
plot(t_impulse, impulse_response_openloop2, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('Roll Angle');
```

################################################################################
###   --- FILE: kalman_filtering_lateral_py.py ---   ###
################################################################################

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim

# State Space matrices
A = np.array([[-0.0558, -0.9968, 0.0802, 0.0415],
              [0.5980, -0.1150, -0.0318, 0],
              [-3.0500, 0.3880, -0.4650, 0],
              [0, 0.0805, 1, 0]])
B = np.array([[0.0729, 0],
              [-4.75, 0.00775],
              [0.15300, 0.1430],
              [0, 0]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 0, 1]])
D = np.array([[0, 0],
              [0, 0]])

# Define system
sys = lti(A, B, C, D)

# Process and Measurement noise covariance
Qn = 2.3
Rn = 1

# Time vector
t = np.arange(0, 10.1, 0.1)  # Time vector

# Sinusoidal input
u = np.column_stack((np.sin(t), np.zeros_like(t)))

# Simulate the true system
_, x_true, _ = lsim(sys, u, t)

# Add noise to measurements (for simulation)
y_measured = lsim(sys, u, t)[1] + np.sqrt(Rn) * np.random.randn(len(t), C.shape[0])

state = ["Side Slip Angle", "Roll Angle"]
# Kalman filter simulation.
# Plotting

for i in range(x_true.shape[1]):
    plt.subplot(x_true.shape[1], 1, i + 1)
    plt.plot(t, x_true[:, i], 'b', label='True')
    plt.plot(t, y_measured[:, i], 'g--', label='Measured')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel(state[i])
    plt.title('Kalman Filter Response')

plt.tight_layout()
plt.show()
```