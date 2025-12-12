"""
Simple anomaly detection method in python.
Identifies points along time series data that deviates from the mean value 
to a highly significant degree. 
"""
from shiny.express import render, ui
from shared import * 

"""
Seaborn config 
"""
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


def server():
    @render.plot(height=300)
    def time_series():

        return 
    
def main():
    # filter by sensor
    selected_sensor = "GT1 W"
    sensor_filtered_df = filtered[filtered['sensor_name'] == selected_sensor]
    if len(sensor_filtered_df) >= 30: # high number of samples 
        print(f'Using Z-score for n > 30')
        sensor_filtered_df['z_score'] = zscore(sensor_filtered_df['value'])
        threshold = 3 # standard for cpaturing highly significant outliers, p<0.001 
        outliers = sensor_filtered_df[(sensor_filtered_df['z_score'].abs() > threshold)]
        # plots trend 
        sns.lineplot(
            x=sensor_filtered_df.index,
            y='value',
            data=sensor_filtered_df,
            label='Normal Data',
            color='blue',
            linewidth=1.5
        )          
        # plot outliers
        sns.scatterplot(
            x=outliers.index,
            y='value',
            data=outliers,
            label='Outlier',
            color='red',
            s=100,
            marker='o'
        )
        plt.title(f'Value over time with Z-score Outliers (threshold > {threshold}) for {selected_sensor}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else: 
        if len(sensor_filtered_df) != 0: 
            print(f'Using Modified Z-score for n < 30')
            threshold = 3.5 #  
            outliers = sensor_filtered_df[np.abs(modified_zscore(sensor_filtered_df['value']) > threshold)]
            sns.lineplot(
                x = sensor_filtered_df.index,
                y = 'value',
                data= sensor_filtered_df,
                label = 'Normal', 
                color = 'blue', 
                linewidth = 1.5
            )
            sns.scatterplot(
                x=outliers.index,
                y='value',
                data=outliers,
                color='red',
                s=100,
                marker='o'
            )
            plt.title(f'Value over time with Outliers for {selected_sensor}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else: 
            print(f'ERROR: No Samples Provided for Sensor {selected_sensor}')

    return print(sensor_filtered_df.head(1))

if __name__ == "__main__":
    main() 