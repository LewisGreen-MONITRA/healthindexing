import numpy as np

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

     return x_upd, P_upd


