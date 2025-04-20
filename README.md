# **Cyber-Physical AI: Anomaly Detection for Drone & Power System Threats**
This project leverages deep LSTM models to detect potential cyber threats in cyber-physical systems, using telemetry and operational data from drones and power grid infrastructures.

This project was developed for a 1-week cyber-physical anomaly detection competition, where participants analyzed data from drones and power systems to identify signs of cyber threats.
Placed 1st overall among participating teams.

## Techniques Used:

- LSTM (Long Short-Term Memory) neural networks to capture temporal patterns and detect subtle anomalies over time

- Sensor fusion of physical and cyber metrics (e.g., GPS, voltage, battery, acceleration, load)

- Preprocessing for noise filtering, normalization, and time-window batching

- Models are trained to detect anomalies indicating potential cyber intrusions, spoofing attacks, or system failures.

## Running

``` python path/to/run/script --data_path /path/to/data --model_path ./path/to/model/model.pth ```
Required packages should be installed from the script if run below command to install dependancies
``` pip install -r requirements.txt ```

Full training and testing can be found in the .ipynb files
