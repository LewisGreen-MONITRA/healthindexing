from shared import * 
from kalman import detectAnomaliesKalman, plotKalmanResults

def main():
    """
    Anomaly detection combining z-score and Kalman filtering approaches.
    Identifies anomalous points in time series data that deviate significantly 
    from expected behavior.
    """
    # Filter by sensor
    filtered = getData()
    selected_sensor = "63-MGC-202 L1"
    sensor_filtered_df = filtered[filtered['sensor_name'] == selected_sensor]
    sensor_filtered_df = sensor_filtered_df[sensor_filtered_df['units'] == "C"]
    
    if len(sensor_filtered_df) == 0:
        print(f'ERROR: No Samples Provided for Sensor {selected_sensor}')
        return
    
    print(f"Analyzing sensor: {selected_sensor}")
    print(f"Total samples: {len(sensor_filtered_df)}")
    print()
    
    # ===== Z-SCORE BASED ANOMALY DETECTION =====
    print("=" * 60)
    print("Z-SCORE BASED ANOMALY DETECTION")
    print("=" * 60)
    
    if len(sensor_filtered_df) >= 30:
        print(f'Using Z-score method (n = {len(sensor_filtered_df)} > 30)')
        sensor_filtered_df['z_score'] = modifiedZscore(sensor_filtered_df['value'])
        threshold = 2  # standard for capturing highly significant outliers, p<0.001
        z_outliers = sensor_filtered_df[(sensor_filtered_df['z_score'].abs() > threshold)]
        print(f"Z-score outliers detected: {len(z_outliers)}")
        
        # Plot z-score results
        plotResults(sensor_filtered_df, z_outliers)
        
        # STL decomposition
        plotSTL(sensor_filtered_df)
        
        # Generalized ESD test
        a_max = 0.1 * len(sensor_filtered_df['value'])
        outlier_index, outlier_value, stats = genESD(
            sensor_filtered_df['value'], 
            alpha=0.005, 
            max=a_max, 
            return_stats=True
        )
        print(f"ESD test detected {len(outlier_index)} outliers")
        print(f"ESD outlier indices: {outlier_index}")
        print()
    else:
        print(f'Using Modified Z-score (n = {len(sensor_filtered_df)} < 30)')
        threshold = 3.5
        z_outliers = sensor_filtered_df[
            np.abs(modifiedZscore(sensor_filtered_df['value'])) > threshold
        ]
        print(f"Modified z-score outliers detected: {len(z_outliers)}")
        plotResults(sensor_filtered_df, z_outliers)
        print()
    
    # ===== KALMAN FILTER BASED ANOMALY DETECTION =====
    print("=" * 60)
    print("KALMAN FILTER + Z-SCORE HYBRID ANOMALY DETECTION")
    print("=" * 60)
    
    # Run Kalman-based hybrid detection
    kalman_results = detectAnomaliesKalman(
        sensor_filtered_df['value'].values,
        z_score_threshold=2.0,
        nis_threshold=6.33,
        window_size=15,
        process_noise=0.01,
        measurement_noise=1e-6
    )
    
    # Add results to dataframe
    sensor_filtered_df['kalman_residual'] = kalman_results['residuals']
    sensor_filtered_df['nis'] = kalman_results['nis_values']
    sensor_filtered_df['kalman_zscore'] = kalman_results['z_scores']
    sensor_filtered_df['is_anomaly'] = kalman_results['anomaly_flags']
    
    n_kalman_anomalies = np.sum(kalman_results['anomaly_flags'])
    print(f"Kalman hybrid method detected: {n_kalman_anomalies} anomalies")
    
    # Show anomalies detected by Kalman filter
    kalman_outliers = sensor_filtered_df[sensor_filtered_df['is_anomaly']]
    if len(kalman_outliers) > 0:
        print(f"\nAnomaly timestamps:")
        for idx, (time, row) in enumerate(kalman_outliers.iterrows(), 1):
            print(f"  {idx}. {time}: value={row['value']:.3e}, "
                  f"residual={row['kalman_residual']:.3e}, "
                  f"NIS={row['nis']:.3f}, z-score={row['kalman_zscore']:.3f}")
    print()
    
    # Plot comprehensive Kalman results
    plotKalmanResults(sensor_filtered_df, kalman_results)
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total data points: {len(sensor_filtered_df)}")
    print(f"Z-score detected anomalies: {len(z_outliers)}")
    print(f"Kalman hybrid detected anomalies: {n_kalman_anomalies}")
    
    # Show data head
    print("\nFirst 5 data points:")
    print(sensor_filtered_df[['value', 'z_score', 'kalman_residual', 
                              'nis', 'kalman_zscore', 'is_anomaly']].head())
    
    return

if __name__ == "__main__":
    main()