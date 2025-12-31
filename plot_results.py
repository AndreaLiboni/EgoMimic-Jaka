import pandas as pd
 
# Read the CSV file
csv_path = '/run/media/Dati/Sviluppo/Università/Tesi/EgoMimic/trained_models_highlevel/pick_bustina/None_DT_2025-12-09-13-21-12/logs/csv_logs/version_0/metrics.csv'
# csv_path = '/run/media/Dati/Sviluppo/Università/Tesi/EgoMimic/trained_models_highlevel/pick_bustina/None_DT_2025-12-27-12-41-55/logs/csv_logs/version_0/metrics.csv'
df = pd.read_csv(csv_path)
 
# Plot the training loss
import matplotlib.pyplot as plt
plt.figure()
plt.plot(df['epoch'], df['Train/Loss'])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')

plt.figure()
plt.plot([p for p in df['Valid/hand_final_mse_avg'] if p is not None and not pd.isna(p)])
plt.plot([p for p in df['Valid/hand_paired_mse_avg'] if p is not None and not pd.isna(p)])
plt.xlabel('Epoch')
plt.ylabel('Hand validation loss')

plt.figure()
plt.plot([p for p in df['Valid/robot_final_mse_avg'] if p is not None and not pd.isna(p)])
plt.plot([p for p in df['Valid/robot_paired_mse_avg'] if p is not None and not pd.isna(p)])
plt.xlabel('Epoch')
plt.ylabel('Robot validation loss')
plt.show()