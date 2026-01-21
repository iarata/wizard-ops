from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Load the stats history
df = pd.read_csv('tests/performancetests/stats/loadtest_stats_history.csv')

# Convert timestamp to relative time (seconds from start)
start_time = df['Timestamp'].min()
df['Time (s)'] = df['Timestamp'] - start_time

# Mark warmup period
warmup_duration = 120
df['Phase'] = df['Time (s)'].apply(lambda x: 'Warmup' if x < warmup_duration else 'Main Test')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Load Test Performance Metrics', fontsize=16, fontweight='bold')

# 1. Response Times over Time
ax1 = axes[0, 0]
ax1.plot(df['Time (s)'], df['50%'], label='p50 (Median)', linewidth=2)
ax1.plot(df['Time (s)'], df['95%'], label='p95', linewidth=2)
ax1.plot(df['Time (s)'], df['99%'], label='p99', linewidth=2)
ax1.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Response Time (ms)')
ax1.set_title('Response Times Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Request Rate over Time
ax2 = axes[0, 1]
ax2.plot(df['Time (s)'], df['Requests/s'], label='Requests/s', color='green', linewidth=2)
ax2.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Requests per Second')
ax2.set_title('Throughput Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. User Count over Time
ax3 = axes[1, 0]
ax3.plot(df['Time (s)'], df['User Count'], label='Active Users', color='blue', linewidth=2)
ax3.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Number of Users')
ax3.set_title('User Load Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.fill_between(df['Time (s)'], 0, df['User Count'], alpha=0.3, color='blue')

# 4. Error Rate over Time
ax4 = axes[1, 1]
ax4.plot(df['Time (s)'], df['Failures/s'], label='Failures/s', color='red', linewidth=2)
ax4.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Failures per Second')
ax4.set_title('Error Rate Over Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
output_dir = Path('tests/performancetests/stats')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'loadtest_visualization.png', dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_dir / 'loadtest_visualization.png'}")

# Show the plot
plt.show()

# Print summary statistics (excluding warmup)
df_no_warmup = df[df['Time (s)'] >= warmup_duration]
print("\n=== Summary Statistics (Excluding Warmup) ===")
print(f"Total Requests: {df_no_warmup['Total Request Count'].max()}")
print(f"Total Failures: {df_no_warmup['Total Failure Count'].max()}")
print(f"Avg Response Time: {df_no_warmup['Total Average Response Time'].mean():.2f} ms")
print(f"Median Response Time: {df_no_warmup['Total Median Response Time'].mean():.2f} ms")
print(f"Max Response Time: {df_no_warmup['Total Max Response Time'].max():.2f} ms")
print(f"Avg Throughput: {df_no_warmup['Requests/s'].mean():.2f} req/s")
print(f"Peak Users: {df_no_warmup['User Count'].max()}")
