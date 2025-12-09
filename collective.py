import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
from scipy.stats import ttest_rel, ttest_ind
import warnings
import os
warnings.filterwarnings('ignore')

# Enhanced styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.3

# Professional color palette
COLORS = {
    'Dashboard 1': '#2E86AB',
    'Dashboard 2': '#A23B72',
    'Fill 1': '#84CAE7',
    'Fill 2': '#F18F01'
}

class CollectiveAnalyzer:
    def __init__(self, base_path='.', confidence_threshold=0.8):
        self.base_path = Path(base_path)
        self.confidence_threshold = confidence_threshold
        self.pupil_data = []
        self.gaze_data = []
        self.tlx_data = []
        self.timing_data = []
        
    def get_user_directories(self):
        user_dirs = [d for d in self.base_path.iterdir() 
                    if d.is_dir() and not d.name.startswith('.') 
                    and not d.name in ['plots', 'output_results']]
        return [d.name for d in user_dirs]
    
    def load_task_timings(self, user_id, dashboard_num):
        csv_path = self.base_path / user_id / f"{user_id}_Dashboard{dashboard_num}_data.csv"
        if not csv_path.exists():
            return None, None
        df = pd.read_csv(csv_path)
        
        # Extract task timings from duration columns
        tasks = {}
        for i in range(1, 4):
            duration_col = f'Task {i} Duration (s)'
            if duration_col in df.columns:
                duration = df[duration_col].iloc[0]
                # Create synthetic start/end times based on duration
                start_time = datetime.now()
                end_time = start_time + timedelta(seconds=duration)
                tasks[f'Task{i}'] = {
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                }
        
        # Return None if no tasks found
        if not tasks:
            return None, None
        
        # Extract NASA-TLX scores
        tlx_scores = {}
        tlx_cols = ['TLX Mental Demand (0-10)', 'TLX Physical Demand (0-10)', 
                'TLX Temporal Demand (0-10)', 'TLX Performance (0-10)', 
                'TLX Effort (0-10)', 'TLX Frustration Level (0-10)']
        for col in tlx_cols:
            if col in df.columns:
                key = col.replace(' (0-10)', '').replace('TLX ', '')
                tlx_scores[key] = df[col].iloc[0]
        
        return tasks, tlx_scores
    
    def load_pupil_data(self, user_id, dashboard_num):
        dashboard_folder = f"{user_id}_D{dashboard_num}"
        pupil_path = self.base_path / user_id / dashboard_folder / 'pupil_positions.csv'
        if not pupil_path.exists():
            return None
        df = pd.read_csv(pupil_path)
        df = df[df['confidence'] >= self.confidence_threshold].copy()
        df_grouped = df.groupby('pupil_timestamp').agg({
            'diameter': 'mean', 'confidence': 'mean'
        }).reset_index()
        df_grouped.rename(columns={'pupil_timestamp': 'timestamp'}, inplace=True)
        return df_grouped
    
    def load_gaze_data(self, user_id, dashboard_num):
        dashboard_folder = f"{user_id}_D{dashboard_num}"
        gaze_path = self.base_path / user_id / dashboard_folder / 'gaze_positions.csv'
        if not gaze_path.exists():
            return None
        df = pd.read_csv(gaze_path)
        df = df[df['confidence'] >= self.confidence_threshold].copy()
        df.rename(columns={'gaze_timestamp': 'timestamp'}, inplace=True)
        return df
    
    def detect_fixations(self, gaze_df, velocity_threshold=30, duration_threshold=100):
        """Detect fixations using velocity threshold"""
        if len(gaze_df) < 2:
            return pd.DataFrame()
        
        gaze_df = gaze_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate velocities
        velocities = [0]
        for i in range(1, len(gaze_df)):
            dx = gaze_df.loc[i, 'norm_pos_x'] - gaze_df.loc[i-1, 'norm_pos_x']
            dy = gaze_df.loc[i, 'norm_pos_y'] - gaze_df.loc[i-1, 'norm_pos_y']
            dt = gaze_df.loc[i, 'timestamp'] - gaze_df.loc[i-1, 'timestamp']
            if dt > 0:
                dist = np.sqrt(dx**2 + dy**2) * 30
                velocity = dist / dt
            else:
                velocity = 0
            velocities.append(velocity)
        
        gaze_df['velocity'] = velocities
        gaze_df['is_fixation'] = gaze_df['velocity'] < velocity_threshold
        gaze_df['fixation_group'] = (gaze_df['is_fixation'] != gaze_df['is_fixation'].shift()).cumsum()
        
        fixations = []
        for group_id, group in gaze_df[gaze_df['is_fixation']].groupby('fixation_group'):
            if len(group) < 3:
                continue
            duration = (group['timestamp'].max() - group['timestamp'].min()) * 1000
            if duration >= duration_threshold:
                fixations.append({
                    'duration_ms': duration,
                    'centroid_x': group['norm_pos_x'].mean(),
                    'centroid_y': group['norm_pos_y'].mean(),
                    'start_time': group['timestamp'].min()
                })
        
        return pd.DataFrame(fixations)
    
    def process_all_users(self):
        """Process all users and collect data"""
        users = self.get_user_directories()
        print(f"\n{'='*80}")
        print(f"Processing {len(users)} users: {users}")
        print(f"{'='*80}\n")
        
        for user_id in users:
            print(f"Processing {user_id}...")
            
            for dashboard_num in [1, 2]:
                dashboard_name = f"Dashboard {dashboard_num}"
                
                # Load task timings and TLX
                tasks, tlx_scores = self.load_task_timings(user_id, dashboard_num)
                if tasks is None:
                    continue
                
                # Store TLX data
                for tlx_key, tlx_value in tlx_scores.items():
                    self.tlx_data.append({
                        'User': user_id,
                        'Dashboard': dashboard_name,
                        'Metric': tlx_key,
                        'Score': tlx_value
                    })
                
                # Store timing data
                for task_name, task_info in tasks.items():
                    self.timing_data.append({
                        'User': user_id,
                        'Dashboard': dashboard_name,
                        'Task': task_name,
                        'Duration': task_info['duration']
                    })
                
                # Load and process pupil data
                pupil_df = self.load_pupil_data(user_id, dashboard_num)
                if pupil_df is not None and len(pupil_df) > 0:
                    # Synchronize timestamps
                    first_task_start = tasks['Task1']['start']
                    recording_start = first_task_start - timedelta(seconds=5)
                    first_timestamp = pupil_df['timestamp'].min()
                    pupil_df['datetime'] = recording_start + pd.to_timedelta(
                        pupil_df['timestamp'] - first_timestamp, unit='s')
                    
                    # Baseline
                    baseline_data = pupil_df[
                        (pupil_df['datetime'] < first_task_start)
                    ]
                    if len(baseline_data) > 0:
                        baseline_mean = baseline_data['diameter'].mean()
                    else:
                        baseline_mean = pupil_df['diameter'].mean()
                    
                    # Process each task
                    for task_name, task_info in tasks.items():
                        task_pupil = pupil_df[
                            (pupil_df['datetime'] >= task_info['start']) & 
                            (pupil_df['datetime'] <= task_info['end'])
                        ].copy()
                        
                        if len(task_pupil) > 0:
                            task_pupil['relative_dilation'] = (
                                (task_pupil['diameter'] - baseline_mean) / baseline_mean
                            )
                            
                            self.pupil_data.append({
                                'User': user_id,
                                'Dashboard': dashboard_name,
                                'Task': task_name,
                                'Mean_Dilation': task_pupil['relative_dilation'].mean(),
                                'Max_Dilation': task_pupil['relative_dilation'].max(),
                                'Std_Dilation': task_pupil['relative_dilation'].std(),
                                'Mean_Diameter': task_pupil['diameter'].mean()
                            })
                
                # Load and process gaze data
                gaze_df = self.load_gaze_data(user_id, dashboard_num)
                if gaze_df is not None and len(gaze_df) > 0:
                    # Synchronize timestamps
                    first_task_start = tasks['Task1']['start']
                    recording_start = first_task_start - timedelta(seconds=5)
                    first_timestamp = gaze_df['timestamp'].min()
                    gaze_df['datetime'] = recording_start + pd.to_timedelta(
                        gaze_df['timestamp'] - first_timestamp, unit='s')
                    
                    # Process each task
                    for task_name, task_info in tasks.items():
                        task_gaze = gaze_df[
                            (gaze_df['datetime'] >= task_info['start']) & 
                            (gaze_df['datetime'] <= task_info['end'])
                        ].copy()
                        
                        if len(task_gaze) > 0:
                            fixations = self.detect_fixations(task_gaze)
                            
                            if len(fixations) > 0:
                                # Calculate scanpath length
                                scanpath_length = 0
                                for i in range(len(fixations) - 1):
                                    dx = fixations.iloc[i+1]['centroid_x'] - fixations.iloc[i]['centroid_x']
                                    dy = fixations.iloc[i+1]['centroid_y'] - fixations.iloc[i]['centroid_y']
                                    scanpath_length += np.sqrt(dx**2 + dy**2)
                                
                                self.gaze_data.append({
                                    'User': user_id,
                                    'Dashboard': dashboard_name,
                                    'Task': task_name,
                                    'Num_Fixations': len(fixations),
                                    'Mean_Fixation_Duration': fixations['duration_ms'].mean(),
                                    'Total_Fixation_Duration': fixations['duration_ms'].sum(),
                                    'Scanpath_Length': scanpath_length
                                })
        
        # Convert to DataFrames
        self.pupil_df = pd.DataFrame(self.pupil_data)
        self.gaze_df = pd.DataFrame(self.gaze_data)
        self.tlx_df = pd.DataFrame(self.tlx_data)
        self.timing_df = pd.DataFrame(self.timing_data)
        
        print(f"\n✅ Data collection complete!")
        print(f"   Pupil records: {len(self.pupil_df)}")
        print(f"   Gaze records: {len(self.gaze_df)}")
        print(f"   TLX records: {len(self.tlx_df)}")
        print(f"   Timing records: {len(self.timing_df)}")
    
    def perform_statistical_tests(self):
        """Perform paired t-tests comparing Dashboard 1 vs Dashboard 2"""
        print(f"\n{'='*80}")
        print("STATISTICAL ANALYSIS - PAIRED T-TESTS")
        print(f"{'='*80}\n")
        
        results = []
        
        # Test 1: Pupil Dilation (Lower is better - less cognitive load)
        if len(self.pupil_df) > 0:
            for task in ['Task1', 'Task2', 'Task3']:
                d1_data = self.pupil_df[
                    (self.pupil_df['Dashboard'] == 'Dashboard 1') & 
                    (self.pupil_df['Task'] == task)
                ]['Mean_Dilation'].values
                
                d2_data = self.pupil_df[
                    (self.pupil_df['Dashboard'] == 'Dashboard 2') & 
                    (self.pupil_df['Task'] == task)
                ]['Mean_Dilation'].values
                
                if len(d1_data) > 0 and len(d2_data) > 0 and len(d1_data) == len(d2_data):
                    t_stat, p_value = ttest_rel(d1_data, d2_data)
                    mean_diff = np.mean(d1_data) - np.mean(d2_data)
                    percent_change = (mean_diff / np.mean(d2_data)) * 100
                    
                    results.append({
                        'Metric': 'Pupil Dilation',
                        'Task': task,
                        'D1_Mean': np.mean(d1_data),
                        'D2_Mean': np.mean(d2_data),
                        'Mean_Difference': mean_diff,
                        'Percent_Change': percent_change,
                        'T_Statistic': t_stat,
                        'P_Value': p_value,
                        'Significant': 'Yes' if p_value < 0.05 else 'No',
                        'Winner': 'Dashboard 1' if mean_diff < 0 else 'Dashboard 2'
                    })
        
        # Test 2: NASA-TLX (Lower is better - less workload)
        if len(self.tlx_df) > 0:
            for metric in self.tlx_df['Metric'].unique():
                d1_data = self.tlx_df[
                    (self.tlx_df['Dashboard'] == 'Dashboard 1') & 
                    (self.tlx_df['Metric'] == metric)
                ]['Score'].values
                
                d2_data = self.tlx_df[
                    (self.tlx_df['Dashboard'] == 'Dashboard 2') & 
                    (self.tlx_df['Metric'] == metric)
                ]['Score'].values
                
                if len(d1_data) > 0 and len(d2_data) > 0 and len(d1_data) == len(d2_data):
                    t_stat, p_value = ttest_rel(d1_data, d2_data)
                    mean_diff = np.mean(d1_data) - np.mean(d2_data)
                    percent_change = (mean_diff / np.mean(d2_data)) * 100
                    
                    results.append({
                        'Metric': f'TLX {metric}',
                        'Task': 'Overall',
                        'D1_Mean': np.mean(d1_data),
                        'D2_Mean': np.mean(d2_data),
                        'Mean_Difference': mean_diff,
                        'Percent_Change': percent_change,
                        'T_Statistic': t_stat,
                        'P_Value': p_value,
                        'Significant': 'Yes' if p_value < 0.05 else 'No',
                        'Winner': 'Dashboard 1' if mean_diff < 0 else 'Dashboard 2'
                    })
        
        # Test 3: Number of Fixations (Fewer is better - more efficient)
        if len(self.gaze_df) > 0:
            for task in ['Task1', 'Task2', 'Task3']:
                d1_data = self.gaze_df[
                    (self.gaze_df['Dashboard'] == 'Dashboard 1') & 
                    (self.gaze_df['Task'] == task)
                ]['Num_Fixations'].values
                
                d2_data = self.gaze_df[
                    (self.gaze_df['Dashboard'] == 'Dashboard 2') & 
                    (self.gaze_df['Task'] == task)
                ]['Num_Fixations'].values
                
                if len(d1_data) > 0 and len(d2_data) > 0 and len(d1_data) == len(d2_data):
                    t_stat, p_value = ttest_rel(d1_data, d2_data)
                    mean_diff = np.mean(d1_data) - np.mean(d2_data)
                    percent_change = (mean_diff / np.mean(d2_data)) * 100
                    
                    results.append({
                        'Metric': 'Num Fixations',
                        'Task': task,
                        'D1_Mean': np.mean(d1_data),
                        'D2_Mean': np.mean(d2_data),
                        'Mean_Difference': mean_diff,
                        'Percent_Change': percent_change,
                        'T_Statistic': t_stat,
                        'P_Value': p_value,
                        'Significant': 'Yes' if p_value < 0.05 else 'No',
                        'Winner': 'Dashboard 1' if mean_diff < 0 else 'Dashboard 2'
                    })
        
        # Test 4: Task Duration (Lower is better - faster completion)
        if len(self.timing_df) > 0:
            for task in ['Task1', 'Task2', 'Task3']:
                d1_data = self.timing_df[
                    (self.timing_df['Dashboard'] == 'Dashboard 1') & 
                    (self.timing_df['Task'] == task)
                ]['Duration'].values
                
                d2_data = self.timing_df[
                    (self.timing_df['Dashboard'] == 'Dashboard 2') & 
                    (self.timing_df['Task'] == task)
                ]['Duration'].values
                
                if len(d1_data) > 0 and len(d2_data) > 0 and len(d1_data) == len(d2_data):
                    t_stat, p_value = ttest_rel(d1_data, d2_data)
                    mean_diff = np.mean(d1_data) - np.mean(d2_data)
                    percent_change = (mean_diff / np.mean(d2_data)) * 100
                    
                    results.append({
                        'Metric': 'Task Duration',
                        'Task': task,
                        'D1_Mean': np.mean(d1_data),
                        'D2_Mean': np.mean(d2_data),
                        'Mean_Difference': mean_diff,
                        'Percent_Change': percent_change,
                        'T_Statistic': t_stat,
                        'P_Value': p_value,
                        'Significant': 'Yes' if p_value < 0.05 else 'No',
                        'Winner': 'Dashboard 1' if mean_diff < 0 else 'Dashboard 2'
                    })
        
        self.test_results = pd.DataFrame(results)
        
        # Display results
        if len(self.test_results) > 0:
            print(self.test_results.to_string(index=False))
            
            # Summary
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            sig_results = self.test_results[self.test_results['Significant'] == 'Yes']
            print(f"\nSignificant results (p < 0.05): {len(sig_results)} / {len(self.test_results)}")
            
            d1_wins = len(self.test_results[self.test_results['Winner'] == 'Dashboard 1'])
            d2_wins = len(self.test_results[self.test_results['Winner'] == 'Dashboard 2'])
            print(f"\nDashboard 1 performed better: {d1_wins} metrics")
            print(f"Dashboard 2 performed better: {d2_wins} metrics")
            
            return self.test_results
        else:
            print("⚠️ No test results generated")
            return pd.DataFrame()
    
    def plot_collective_pupil_dilation(self, save_path=None):
        """Plot collective pupil dilation across all tasks"""
        if len(self.pupil_df) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.patch.set_facecolor('white')
        
        tasks = ['Task1', 'Task2', 'Task3']
        task_labels = ['Task 1', 'Task 2', 'Task 3']
        
        for idx, (task, label) in enumerate(zip(tasks, task_labels)):
            ax = axes[idx]
            ax.set_facecolor('#fafafa')
            
            for dashboard in ['Dashboard 1', 'Dashboard 2']:
                data = self.pupil_df[
                    (self.pupil_df['Dashboard'] == dashboard) & 
                    (self.pupil_df['Task'] == task)
                ]['Mean_Dilation'].values
                
                if len(data) > 0:
                    color = COLORS[dashboard]
                    positions = [idx * 2 + (0 if dashboard == 'Dashboard 1' else 1)]
                    
                    # Box plot
                    bp = ax.boxplot([data], positions=positions, widths=0.6,
                                   patch_artist=True, showmeans=True,
                                   meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                                   boxprops=dict(linewidth=2.5),
                                   whiskerprops=dict(linewidth=2),
                                   capprops=dict(linewidth=2),
                                   medianprops=dict(linewidth=3, color='darkred'))
                    
                    for patch in bp['boxes']:
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Individual points
                    x_pos = positions[0] + np.random.normal(0, 0.04, len(data))
                    ax.scatter(x_pos, data, alpha=0.6, s=100, color=color, 
                              edgecolors='white', linewidth=2, zorder=10)
                    
                    # Mean line
                    mean_val = np.mean(data)
                    ax.hlines(mean_val, positions[0] - 0.3, positions[0] + 0.3,
                             colors=color, linewidth=3, linestyles='--', alpha=0.8)
            
            ax.axhline(y=0, color='#636363', linestyle='-', linewidth=2, alpha=0.5)
            ax.set_xticks([idx * 2, idx * 2 + 1])
            ax.set_xticklabels(['D1', 'D2'], fontsize=13, fontweight='bold')
            ax.set_ylabel('Relative Pupil Dilation', fontsize=14, fontweight='bold', color='#2c3e50')
            ax.set_title(label, fontsize=16, fontweight='bold', pad=15, color='#2c3e50')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_linewidth(2)
        
        plt.suptitle('Collective Pupil Dilation Analysis (All Users)', 
                    fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {save_path}")
        plt.show()
    
    def plot_collective_nasa_tlx(self, save_path=None):
        """Plot collective NASA-TLX scores"""
        if len(self.tlx_df) == 0:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        
        metrics = self.tlx_df['Metric'].unique()
        x = np.arange(len(metrics))
        width = 0.35
        
        d1_means = []
        d1_stds = []
        d2_means = []
        d2_stds = []
        
        for metric in metrics:
            d1 = self.tlx_df[
                (self.tlx_df['Dashboard'] == 'Dashboard 1') & 
                (self.tlx_df['Metric'] == metric)
            ]['Score'].values
            
            d2 = self.tlx_df[
                (self.tlx_df['Dashboard'] == 'Dashboard 2') & 
                (self.tlx_df['Metric'] == metric)
            ]['Score'].values
            
            d1_means.append(np.mean(d1) if len(d1) > 0 else 0)
            d1_stds.append(np.std(d1) if len(d1) > 0 else 0)
            d2_means.append(np.mean(d2) if len(d2) > 0 else 0)
            d2_stds.append(np.std(d2) if len(d2) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, d1_means, width, label='Dashboard 1',
                      color=COLORS['Dashboard 1'], alpha=0.85, edgecolor='white', 
                      linewidth=3, yerr=d1_stds, capsize=8,
                      error_kw={'linewidth': 2.5, 'ecolor': '#34495e'})
        
        bars2 = ax.bar(x + width/2, d2_means, width, label='Dashboard 2',
                      color=COLORS['Dashboard 2'], alpha=0.85, edgecolor='white', 
                      linewidth=3, yerr=d2_stds, capsize=8,
                      error_kw={'linewidth': 2.5, 'ecolor': '#34495e'})
        
        # Add value labels
        for bars, means in [(bars1, d1_means), (bars2, d2_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                       f'{mean:.2f}', ha='center', va='bottom', 
                       fontsize=11, fontweight='bold', color='#2c3e50')
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (0-10, Lower is Better)', fontsize=14, 
                     fontweight='bold', color='#2c3e50')
        ax.set_title('NASA-TLX Workload Assessment (All Users)', 
                    fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
        ax.legend(fontsize=13, framealpha=0.98, loc='upper right', 
                 edgecolor='#95a5a6', fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, max(max(d1_means), max(d2_means)) * 1.2)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(2)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {save_path}")
        plt.show()
    
    def plot_collective_gaze_metrics(self, save_path=None):
        """Plot collective gaze metrics"""
        if len(self.gaze_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.patch.set_facecolor('white')
        axes = axes.flatten()
        
        metrics = [
            ('Num_Fixations', 'Number of Fixations', 'Lower is Better'),
            ('Mean_Fixation_Duration', 'Mean Fixation Duration (ms)', 'Contextual'),
            ('Total_Fixation_Duration', 'Total Fixation Time (ms)', 'Lower is Better'),
            ('Scanpath_Length', 'Scanpath Length', 'Lower is Better')
        ]
        
        for idx, (metric, title, interpretation) in enumerate(metrics):
            ax = axes[idx]
            ax.set_facecolor('#fafafa')
            
            tasks = ['Task1', 'Task2', 'Task3']
            x = np.arange(len(tasks))
            width = 0.35
            
            d1_means = []
            d2_means = []
            
            for task in tasks:
                d1 = self.gaze_df[
                    (self.gaze_df['Dashboard'] == 'Dashboard 1') & 
                    (self.gaze_df['Task'] == task)
                ][metric].values
                
                d2 = self.gaze_df[
                    (self.gaze_df['Dashboard'] == 'Dashboard 2') & 
                    (self.gaze_df['Task'] == task)
                ][metric].values
                
                d1_means.append(np.mean(d1) if len(d1) > 0 else 0)
                d2_means.append(np.mean(d2) if len(d2) > 0 else 0)
            
            bars1 = ax.bar(x - width/2, d1_means, width, label='Dashboard 1',
                          color=COLORS['Dashboard 1'], alpha=0.8, 
                          edgecolor='white', linewidth=2)
            bars2 = ax.bar(x + width/2, d2_means, width, label='Dashboard 2',
                          color=COLORS['Dashboard 2'], alpha=0.8, 
                          edgecolor='white', linewidth=2)
            
            # Add value labels
            for bars, means in [(bars1, d1_means), (bars2, d2_means)]:
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.1f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color='#2c3e50')
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3'], fontsize=11)
            ax.set_ylabel(title, fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_title(f'{title}\n({interpretation})', fontsize=13, 
                        fontweight='bold', pad=12, color='#2c3e50')
            
            if idx == 0:
                ax.legend(fontsize=11, framealpha=0.98)
            
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_linewidth(2)
        
        plt.suptitle('Collective Gaze Analysis (All Users)', 
                    fontsize=20, fontweight='bold', y=0.995, color='#2c3e50')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {save_path}")
        plt.show()
    
    def plot_collective_timing(self, save_path=None):
        """Plot task completion times"""
        if len(self.timing_df) == 0:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        
        tasks = ['Task1', 'Task2', 'Task3']
        x = np.arange(len(tasks))
        width = 0.35
        
        d1_means = []
        d1_stds = []
        d2_means = []
        d2_stds = []
        
        for task in tasks:
            d1 = self.timing_df[
                (self.timing_df['Dashboard'] == 'Dashboard 1') & 
                (self.timing_df['Task'] == task)
            ]['Duration'].values
            
            d2 = self.timing_df[
                (self.timing_df['Dashboard'] == 'Dashboard 2') & 
                (self.timing_df['Task'] == task)
            ]['Duration'].values
            
            d1_means.append(np.mean(d1) if len(d1) > 0 else 0)
            d1_stds.append(np.std(d1) if len(d1) > 0 else 0)
            d2_means.append(np.mean(d2) if len(d2) > 0 else 0)
            d2_stds.append(np.std(d2) if len(d2) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, d1_means, width, label='Dashboard 1',
                      color=COLORS['Dashboard 1'], alpha=0.85, edgecolor='white', 
                      linewidth=3, yerr=d1_stds, capsize=10,
                      error_kw={'linewidth': 2.5, 'ecolor': '#34495e'})
        
        bars2 = ax.bar(x + width/2, d2_means, width, label='Dashboard 2',
                      color=COLORS['Dashboard 2'], alpha=0.85, edgecolor='white', 
                      linewidth=3, yerr=d2_stds, capsize=10,
                      error_kw={'linewidth': 2.5, 'ecolor': '#34495e'})
        
        # Add value labels
        for bars, means in [(bars1, d1_means), (bars2, d2_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(d1_stds + d2_stds) * 0.5,
                       f'{mean:.1f}s', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold', color='#2c3e50')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3'], fontsize=13, fontweight='bold')
        ax.set_ylabel('Duration (seconds, Lower is Better)', fontsize=14, 
                     fontweight='bold', color='#2c3e50')
        ax.set_title('Task Completion Times (All Users)', 
                    fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
        ax.legend(fontsize=13, framealpha=0.98, loc='upper right',
                 edgecolor='#95a5a6', fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(2)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {save_path}")
        plt.show()
    
    def plot_statistical_results(self, save_path=None):
        """Visualize t-test results"""
        if not hasattr(self, 'test_results') or len(self.test_results) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.patch.set_facecolor('white')
        
        # Plot 1: P-values heatmap
        ax = axes[0, 0]
        ax.set_facecolor('#fafafa')
        
        # Prepare data for heatmap
        metrics = self.test_results['Metric'].unique()
        tasks = self.test_results['Task'].unique()
        
        p_matrix = np.zeros((len(metrics), len(tasks)))
        for i, metric in enumerate(metrics):
            for j, task in enumerate(tasks):
                row = self.test_results[
                    (self.test_results['Metric'] == metric) & 
                    (self.test_results['Task'] == task)
                ]
                if len(row) > 0:
                    p_matrix[i, j] = row.iloc[0]['P_Value']
                else:
                    p_matrix[i, j] = 1.0
        
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(tasks, fontsize=11)
        ax.set_yticklabels(metrics, fontsize=11)
        
        # Add p-values as text
        for i in range(len(metrics)):
            for j in range(len(tasks)):
                text = ax.text(j, i, f'{p_matrix[i, j]:.4f}',
                             ha="center", va="center", color="black", 
                             fontsize=10, fontweight='bold')
        
        ax.set_title('P-Values (Darker = More Significant)', 
                    fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        plt.colorbar(im, ax=ax, label='P-Value')
        
        # Plot 2: Percent change
        ax = axes[0, 1]
        ax.set_facecolor('#fafafa')
        
        sig_results = self.test_results[self.test_results['Significant'] == 'Yes']
        if len(sig_results) > 0:
            y_pos = np.arange(len(sig_results))
            colors = [COLORS['Dashboard 1'] if x < 0 else COLORS['Dashboard 2'] 
                     for x in sig_results['Percent_Change']]
            
            bars = ax.barh(y_pos, sig_results['Percent_Change'], 
                          color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{row['Metric']}\n{row['Task']}" 
                               for _, row in sig_results.iterrows()], 
                              fontsize=10)
            ax.set_xlabel('Percent Change (D1 vs D2)', fontsize=12, fontweight='bold')
            ax.set_title('Significant Improvements\n(Negative = D1 Better)', 
                        fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
        
        # Plot 3: Winner summary
        ax = axes[1, 0]
        ax.set_facecolor('#fafafa')
        
        winner_counts = self.test_results['Winner'].value_counts()
        colors_pie = [COLORS['Dashboard 1'] if x == 'Dashboard 1' else COLORS['Dashboard 2'] 
                     for x in winner_counts.index]
        
        wedges, texts, autotexts = ax.pie(winner_counts.values, labels=winner_counts.index,
                                          autopct='%1.1f%%', colors=colors_pie, 
                                          startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
        ax.set_title('Overall Performance Winner\n(Across All Metrics)', 
                    fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        
        # Plot 4: Significance summary
        ax = axes[1, 1]
        ax.set_facecolor('#fafafa')
        
        sig_counts = self.test_results['Significant'].value_counts()
        colors_bar = ['#27ae60' if x == 'Yes' else '#e74c3c' for x in sig_counts.index]
        
        bars = ax.bar(sig_counts.index, sig_counts.values, 
                     color=colors_bar, alpha=0.8, edgecolor='white', linewidth=3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold', color='#2c3e50')
        
        ax.set_ylabel('Number of Tests', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Significance Summary\n(p < 0.05)', 
                    fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.suptitle('Statistical Analysis Results', 
                    fontsize=20, fontweight='bold', y=0.995, color='#2c3e50')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {save_path}")
        plt.show()
    
    def generate_performance_report(self, output_path='output_results/performance_report.txt'):
        """Generate comprehensive performance report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DASHBOARD PERFORMANCE COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            
            if hasattr(self, 'test_results') and len(self.test_results) > 0:
                d1_wins = len(self.test_results[self.test_results['Winner'] == 'Dashboard 1'])
                d2_wins = len(self.test_results[self.test_results['Winner'] == 'Dashboard 2'])
                sig_results = len(self.test_results[self.test_results['Significant'] == 'Yes'])
                
                f.write(f"Total metrics analyzed: {len(self.test_results)}\n")
                f.write(f"Statistically significant results: {sig_results} ({sig_results/len(self.test_results)*100:.1f}%)\n")
                f.write(f"Dashboard 1 performed better: {d1_wins} metrics ({d1_wins/len(self.test_results)*100:.1f}%)\n")
                f.write(f"Dashboard 2 performed better: {d2_wins} metrics ({d2_wins/len(self.test_results)*100:.1f}%)\n\n")
                
                # Detailed results by category
                f.write("DETAILED RESULTS BY CATEGORY\n")
                f.write("-"*80 + "\n\n")
                
                categories = ['Pupil Dilation', 'TLX', 'Num Fixations', 'Task Duration']
                for category in categories:
                    cat_results = self.test_results[self.test_results['Metric'].str.contains(category)]
                    if len(cat_results) > 0:
                        f.write(f"\n{category.upper()}\n")
                        f.write("-"*40 + "\n")
                        for _, row in cat_results.iterrows():
                            f.write(f"  {row['Task']}: ")
                            f.write(f"D1={row['D1_Mean']:.4f}, D2={row['D2_Mean']:.4f}, ")
                            f.write(f"Change={row['Percent_Change']:.2f}%, ")
                            f.write(f"p={row['P_Value']:.4f} ")
                            f.write(f"{'***' if row['Significant'] == 'Yes' else ''}\n")
                            f.write(f"    Winner: {row['Winner']}\n")
                
                # Key findings
                f.write("\n\nKEY FINDINGS\n")
                f.write("-"*80 + "\n")
                
                sig_improvements = self.test_results[
                    (self.test_results['Significant'] == 'Yes') & 
                    (self.test_results['Winner'] == 'Dashboard 1')
                ]
                
                if len(sig_improvements) > 0:
                    f.write("\nSignificant improvements in Dashboard 1:\n")
                    for _, row in sig_improvements.iterrows():
                        f.write(f"  • {row['Metric']} ({row['Task']}): {abs(row['Percent_Change']):.2f}% improvement\n")
                
                sig_regressions = self.test_results[
                    (self.test_results['Significant'] == 'Yes') & 
                    (self.test_results['Winner'] == 'Dashboard 2')
                ]
                
                if len(sig_regressions) > 0:
                    f.write("\nAreas where Dashboard 2 performed better:\n")
                    for _, row in sig_regressions.iterrows():
                        f.write(f"  • {row['Metric']} ({row['Task']}): {abs(row['Percent_Change']):.2f}% better\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✅ Performance report saved: {output_path}")
    
    def export_all_results(self, output_dir='output_results'):
        """Export all data and results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.pupil_df) > 0:
            self.pupil_df.to_csv(f'{output_dir}/pupil_collective.csv', index=False)
            print(f"✅ Exported: {output_dir}/pupil_collective.csv")
        
        if len(self.gaze_df) > 0:
            self.gaze_df.to_csv(f'{output_dir}/gaze_collective.csv', index=False)
            print(f"✅ Exported: {output_dir}/gaze_collective.csv")
        
        if len(self.tlx_df) > 0:
            self.tlx_df.to_csv(f'{output_dir}/tlx_collective.csv', index=False)
            print(f"✅ Exported: {output_dir}/tlx_collective.csv")
        
        if len(self.timing_df) > 0:
            self.timing_df.to_csv(f'{output_dir}/timing_collective.csv', index=False)
            print(f"✅ Exported: {output_dir}/timing_collective.csv")
        
        if hasattr(self, 'test_results') and len(self.test_results) > 0:
            self.test_results.to_csv(f'{output_dir}/statistical_tests.csv', index=False)
            print(f"✅ Exported: {output_dir}/statistical_tests.csv")


# MAIN EXECUTION
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COLLECTIVE DASHBOARD ANALYSIS WITH STATISTICAL TESTING")
    print("="*80 + "\n")
    
    analyzer = CollectiveAnalyzer(base_path='.', confidence_threshold=0.8)
    
    # Step 1: Process all users
    analyzer.process_all_users()
    
    # Step 2: Perform statistical tests
    print("\n" + "="*80)
    test_results = analyzer.perform_statistical_tests()
    
    # Step 3: Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    os.makedirs('plots/collective', exist_ok=True)
    
    print("📊 Plotting pupil dilation...")
    analyzer.plot_collective_pupil_dilation('plots/collective/pupil_dilation.png')
    
    print("📊 Plotting NASA-TLX scores...")
    analyzer.plot_collective_nasa_tlx('plots/collective/nasa_tlx.png')
    
    print("📊 Plotting gaze metrics...")
    analyzer.plot_collective_gaze_metrics('plots/collective/gaze_metrics.png')
    
    print("📊 Plotting task timings...")
    analyzer.plot_collective_timing('plots/collective/task_timing.png')
    
    print("📊 Plotting statistical results...")
    analyzer.plot_statistical_results('plots/collective/statistical_results.png')
    
    # Step 4: Export results
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80 + "\n")
    
    analyzer.export_all_results('output_results')
    analyzer.generate_performance_report('output_results/performance_report.txt')
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📁 Output locations:")
    print("   📊 Plots: ./plots/collective/")
    print("   📈 Data: ./output_results/")
    print("   📄 Report: ./output_results/performance_report.txt")
    print("\n" + "="*80)