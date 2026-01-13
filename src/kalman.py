import numpy as np
import shared
import matplotlib.pyplot as plt

# time step
dt = 1.0  

#  transition matrix
A = np.array([[1.0, dt],
              [0.0, 1.0]])  

# observation matrix
H = np.array([[1.0, 0.0]])  

# process noise 
Q = np.array([[1.0, 0.0],
              [0.0, 1.0]])  

# measurement noise
R = np.array([[1e-2]])  

# initial state estimation 
x = np.array([[0.0], [1.0]])  

# initial uncertainty
P = np.array([[1.0, 0.0],
              [0.0, 1.0]])  


def kalmanFilter(x, P, z, A, H, Q, R):
     """
     Simple Kalman Filter Implementation
     Approach for covariance estimation

     :return: updated state and uncertainty

     """
     x_pred = A @ x

     P_pred = A @ P @ A.T + Q

     z = np.array([[0.5]])

     y = z - H @ x_pred
     S = H @ P_pred @ H.T + R
     K = P_pred @ H.T @ np.linalg.inv(S)

     x_upd = x_pred + K @ y

     I = np.eye(P.shape[0])
     P_upd = (I - K @ H) @ P_pred

     nis = float(y.T @ np.linalg.inv(S) @ y)

     return x_upd, P_upd, y.item(), nis


# example with filtered data, given sensor and unit 
df = shared.getData()
df = df[df['sensor_name'] == "63-MGC-202 L1"]
df = df[df['units'] == "C"]
residuals = []
nis_values = []
z_scores = []

hybrid_flags = []  


for value in df['value']:
     x = np.array([[value], [0.0]])
     x, P, residual, nis = kalmanFilter(x, P, value, A, H, Q, R)
     print(f'Updated State: {x.flatten()}, Uncertainty: {P}, Residual: {residual}, NIS: {nis}')
    
     residuals.append(residual)
     nis_values.append(nis)

     if len(residuals) >= 15:
          window = residuals[-15:]
          mu = np.mean(window)
          sigma = np.std(window) + 1e-8
          z_score = (residual - mu) / sigma
     else:
          z_score = 0.0

z_scores.append(z_score)
hybrid_flags.append(
     (nis > 6.63) or (abs(z_score) > 3.0)
)

plt.figure()
plt.plot(df['value'].values, label="Measurement")
plt.plot(hybrid_flags, label="Hybrid anomaly flag")
plt.xlabel("Time step")
plt.ylabel("Value / Flag")
plt.legend()
plt.title("Hybrid Kalman + Z-score Anomaly Detector")
plt.show()





"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Kalman filter setup
# -----------------------------
dt = 1.0

A = np.array([[1.0, dt],
              [0.0, 1.0]])

H = np.array([[1.0, 0.0]])

Q = np.array([[1e-4, 0.0],
              [0.0, 1e-6]])

R = np.array([[1e-2]])

x = np.array([[0.0],
              [0.0]])

P = np.eye(2)

def kalman_step_with_nis(x, P, z, A, H, Q, R):
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_new = x_pred + K @ y
    P_new = (np.eye(len(x)) - K @ H) @ P_pred

    nis = float(y.T @ np.linalg.inv(S) @ y)

    return x_new, P_new, y.item(), nis

# -----------------------------
# Simulated data with anomalies
# -----------------------------
np.random.seed(1)
N = 100
true_voltage = 1.0
noise = 0.1 * np.random.randn(N)

measurements = true_voltage + noise

# Inject anomalies
measurements[40] += 0.8     # spike
measurements[70:75] += 0.4  # sustained bias

# -----------------------------
# Hybrid detector
# -----------------------------
residuals = []
nis_values = []
z_scores = []

z_window = 15
z_threshold = 3.0
nis_threshold = 6.63  # chi-square 99% for 1D

hybrid_flags = []

for z in measurements:
    x, P, residual, nis = kalman_step_with_nis(
        x, P, np.array([[z]]), A, H, Q, R
    )

    residuals.append(residual)
    nis_values.append(nis)

    if len(residuals) >= z_window:
        window = residuals[-z_window:]
        mu = np.mean(window)
        sigma = np.std(window) + 1e-8
        z_score = (residual - mu) / sigma
    else:
        z_score = 0.0

    z_scores.append(z_score)

    hybrid_flags.append(
        (nis > nis_threshold) or (abs(z_score) > z_threshold)
    )

# -----------------------------
# Visualization
# -----------------------------
plt.figure()
plt.plot(measurements, label="Measurement")
plt.plot(hybrid_flags, label="Hybrid anomaly flag")
plt.xlabel("Time step")
plt.ylabel("Value / Flag")
plt.legend()
plt.title("Hybrid Kalman + Z-score Anomaly Detector")
plt.show()

"""