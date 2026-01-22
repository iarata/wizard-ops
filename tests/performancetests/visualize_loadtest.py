from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from typer import Typer

app = Typer()

def plot_response_times(ax, df, warmup_duration = None):
    """Plots 50, 95 and 99% quantiles from a loadtest df."""
    ax.plot(df['Time (s)'], df['50%']/1000, label='p50 (Median)', linewidth=2)
    ax.plot(df['Time (s)'], df['95%']/1000, label='p95', linewidth=2)
    ax.plot(df['Time (s)'], df['99%']/1000, label='p99', linewidth=2)

    if warmup_duration is not None:
        ax.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Response Time (s)')
    ax.set_title('Response Times Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_requests_per_sec(ax, df, warmup_duration = None):
    ax.plot(df['Time (s)'], df['Requests/s'], label='Requests/s', color='green', linewidth=2)
    if warmup_duration is not None:
        ax.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Requests per Second')
    ax.set_title('Throughput Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_user_count(ax, df, warmup_duration = None):
    ax.plot(df['Time (s)'], df['User Count'], label='Active Users', color='blue', linewidth=2)
    
    if warmup_duration is not None:
        ax.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of Users')
    ax.set_title('User Load Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.fill_between(df['Time (s)'], 0, df['User Count'], alpha=0.3, color='blue')

def plot_failuers_per_sec(ax , df, warmup_duration = None):
    ax.plot(df['Time (s)'], df['Failures/s'], label='Failures/s', color='red', linewidth=2)
    
    if warmup_duration is not None:
        ax.axvline(x=warmup_duration, color='red', linestyle='--', label='Warmup End', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Failures per Second')
    ax.set_title('Error Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

@app.command()
def plot_single(test_history_path='tests/performancetests/stats/loadtest_stats_history.csv', output_dir = 'tests/performancetests/stats'):
    # Load the stats history
    df = pd.read_csv(test_history_path)

    # Convert timestamp to relative time (seconds from start)
    start_time = df['Timestamp'].min()
    df['Time (s)'] = df['Timestamp'] - start_time

    # Mark warmup period
    warmup_duration = 120

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Load Test Performance Metrics', fontsize=16, fontweight='bold')

    # 1. Response Times over Time
    ax1 = axes[0, 0]
    plot_response_times(ax=ax1, df=df, warmup_duration=warmup_duration)
    

    # 2. Request Rate over Time
    ax2 = axes[0, 1]
    plot_requests_per_sec(ax=ax2, df=df, warmup_duration=warmup_duration)

    # 3. User Count over Time
    ax3 = axes[1, 0]
    plot_user_count(ax=ax3, df=df, warmup_duration=warmup_duration)

    # 4. Error Rate over Time
    ax4 = axes[1, 1]
    plot_failuers_per_sec(ax=ax4, df=df, warmup_duration=warmup_duration)

    plt.tight_layout()

    # Save the figure
    output_dir = Path(output_dir)
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


@app.command()
def plot_average(test_dir: str = 'tests/performancetests/stats', output_dir: str = 'tests/performancetests/stats', num_tests: int =3):
    df_list = []
    for i in range(num_tests):
        df = pd.read_csv(f'{test_dir}/{i+1}/loadtest_stats_history.csv')
        # Convert timestamp to relative time (seconds from start)
        start_time = df['Timestamp'].min()
        df['Time (s)'] = df['Timestamp'] - start_time
        df_list.append(df)

    # Concatenate all dataframes and group by Time to average across runs
    df_concat = pd.concat(df_list, ignore_index=True)
    df_means = df_concat.groupby('Time (s)').mean(numeric_only=True).reset_index()
    
    # Mark warmup period
    warmup_duration = 120

    df['Phase'] = df['Time (s)'].apply(lambda x: 'Warmup' if x < warmup_duration else 'Main Test')

    df_means['Phase'] = df_means['Time (s)'].apply(lambda x: 'Warmup' if x < warmup_duration else 'Main Test')

    # Create individual plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Response Times over Time
    fig, ax = plt.subplots(1, figsize=(7, 5))
    plot_response_times(ax=ax, df=df_means, warmup_duration=warmup_duration)
    plt.savefig(output_dir / 'loadtest_response_times.png', dpi=300, bbox_inches='tight')


    # 2. Request Rate and Error Rate over Time
    fig, ax = plt.subplots(1, figsize=(7, 5))
    plot_requests_per_sec(ax=ax, df=df_means, warmup_duration=warmup_duration)
    plot_failuers_per_sec(ax=ax, df=df_means)
    ax.set_title("Requests and failures per second.")
    plt.savefig(output_dir / 'loadtest_request_failures_per_sec.png', dpi=300, bbox_inches='tight')

    # 3. User Count over Time
    fig, ax = plt.subplots(1, figsize=(7, 5))
    plot_user_count(ax=ax, df=df_means, warmup_duration=warmup_duration)
    plt.savefig(output_dir / 'loadtest_user_count.png', dpi=300, bbox_inches='tight')

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

if __name__ == "__main__":
    app()