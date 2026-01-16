"""
Kalman Filter Implementation for Time Series Anomaly Detection

Combines Kalman filtering with z-score calculations to detect anomalous events
in sensor data time series.
"""

import numpy as np
from shared import *
import matplotlib.pyplot as plt


class KalmanFilterAnomalyDetector:
    """
    Kalman filter-based anomaly detector for time series data.
    Combines residual-based detection with z-score analysis.
    """
    
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=0.01):
        """
        Initialize Kalman filter parameters.
        
        :param dt: time step
        :param process_noise: process noise covariance (Q)
        :param measurement_noise: measurement noise covariance (R)
        """
        self.dt = dt
        
        # Transition matrix (constant velocity model)
        self.A = np.array([[1.0, dt],
                          [0.0, 1.0]])
        
        # Observation matrix (we observe position only)
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.Q = np.array([[process_noise, 0.0],
                          [0.0, process_noise]])
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])
        
        # Initial state [position, velocity]
        self.x = np.array([[0.0], [0.0]])
        
        # Initial state covariance
        self.P = np.array([[1.0, 0.0],
                          [0.0, 1.0]])
    
    def predict(self):
        """
        Prediction step of Kalman filter.
        
        :return: predicted state and covariance
        """
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        return x_pred, P_pred
    
    def update(self, z, x_pred, P_pred):
        """
        Update step of Kalman filter.
        
        :param z: measurement (scalar)
        :param x_pred: predicted state
        :param P_pred: predicted covariance
        :return: updated state, covariance, residual, and NIS
        """
        z = np.array([[z]])
        
        # Innovation (residual)
        y = z - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Updated state
        x_upd = x_pred + K @ y
        
        # Updated covariance
        I = np.eye(self.P.shape[0])
        P_upd = (I - K @ self.H) @ P_pred
        
        # Normalized Innovation Squared (NIS)
        nis = float((y.T @ np.linalg.inv(S) @ y).item())
        
        return x_upd, P_upd, y.item(), nis
    
    def filter(self, z):
        """
        Single filter step (predict + update).
        
        :param z: measurement value
        :return: residual and NIS
        """
        x_pred, P_pred = self.predict()
        self.x, self.P, residual, nis = self.update(z, x_pred, P_pred)
        return residual, nis
    
    def reset(self):
        """Reset filter state."""
        self.x = np.array([[0.0], [0.0]])
        self.P = np.array([[1.0, 0.0],
                          [0.0, 1.0]])


def detectAnomaliesKalman(time_series, z_score_threshold=2.0, nis_threshold=6.33, 
                          window_size=15, process_noise=0.01, measurement_noise=1e-6):
    """
    Detect anomalies using hybrid Kalman + Z-score approach.
    
    :param time_series: array of sensor values
    :param z_score_threshold: threshold for z-score based detection
    :param nis_threshold: threshold for NIS-based detection
    :param window_size: window size for rolling z-score calculation
    :param process_noise: Kalman filter process noise
    :param measurement_noise: Kalman filter measurement noise
    :return: dict with residuals, NIS values, z-scores, and anomaly flags
    """
    
    kf = KalmanFilterAnomalyDetector(process_noise=process_noise, 
                                    measurement_noise=measurement_noise)
    
    residuals = []
    nis_values = []
    z_scores = []
    anomaly_flags = []
    
    for value in time_series:
        residual, nis = kf.filter(float(value))
        
        residuals.append(residual)
        nis_values.append(nis)
        
        # Calculate rolling z-score on residuals
        if len(residuals) >= window_size:
            window = residuals[-window_size:]
            mu = np.mean(window)
            sigma = np.std(window) + 1e-15  # prevent division by zero
            if sigma > 1e-15:
                z_score = (residual - mu) / sigma
            else:
                z_score = 0.0
        else:
            z_score = 0.0
        
        z_scores.append(z_score)
            
        # Hybrid anomaly detection: NIS-based OR z-score based
        is_anomaly = (nis > nis_threshold) or (np.abs(z_score) > z_score_threshold)
        anomaly_flags.append(is_anomaly)
    
    return {
        'residuals': np.array(residuals),
        'nis_values': np.array(nis_values),
        'z_scores': np.array(z_scores),
        'anomaly_flags': np.array(anomaly_flags)
    }


def plotKalmanResults(df, detection_results):
    """
    Plot Kalman filter anomaly detection results.
    
    :param df: dataframe with time index and values
    :param detection_results: dict from detectAnomaliesKalman()
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time_index = range(len(df))
    units = str(df['units'].unique())
    
    # Plot 1: Measurements with anomalies highlighted
    axes[0].plot(time_index, df['value'].values, label='Sensor Value', 
                 color='blue', linewidth=1.5, alpha=0.7)
    anomaly_indices = np.where(detection_results['anomaly_flags'])[0]
    if len(anomaly_indices) > 0:
        axes[0].scatter(anomaly_indices, df['value'].values[anomaly_indices], 
                       color='red', s=100, label='Anomaly', marker='o', zorder=5)
    axes[0].set_ylabel(units)
    axes[0].set_title('Time Series with Detected Anomalies')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals and Z-scores
    axes[1].plot(time_index, detection_results['residuals'], label='Residuals', 
                 color='green', alpha=0.7)
    axes[1].plot(time_index, detection_results['z_scores'], label='Z-scores', 
                 color='orange', alpha=0.7)
    axes[1].axhline(y=2.0, color='r', linestyle='--', label='Z-score threshold')
    axes[1].axhline(y=-2.0, color='r', linestyle='--')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Kalman Residuals and Z-scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: NIS values
    axes[2].plot(time_index, detection_results['nis_values'], label='NIS', 
                 color='purple', alpha=0.7)
    axes[2].axhline(y=6.33, color='r', linestyle='--', label='NIS threshold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('NIS')
    axes[2].set_title('Normalsed Innovation Squared (NIS)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with sensor data
    df = getData()
    selected_sensor = "63-MGC-202 L1"
    sensor_df = df[df['sensor_name'] == selected_sensor]
    sensor_df = sensor_df[sensor_df['units'] == "C"]
    
    if len(sensor_df) > 0:
        print(f"Analyzing {selected_sensor} with {len(sensor_df)} samples")
        
        # Run Kalman-based anomaly detection with tuned parameters
        results = detectAnomaliesKalman(
            sensor_df['value'].values,
            z_score_threshold=2.0,
            nis_threshold=6.33,
            window_size=15,
            process_noise=0.01,
            measurement_noise=1e-6
        )
        
        # Print summary
        n_anomalies = np.sum(results['anomaly_flags'])
        print(f"Detected {n_anomalies} anomalies")
        
        # Plot results
        plotKalmanResults(sensor_df, results)
    else:
        print(f"No data found for sensor {selected_sensor}")

