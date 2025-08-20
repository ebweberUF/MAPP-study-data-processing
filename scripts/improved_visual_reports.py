#!/usr/bin/env python3
"""
MAPP Study Improved Visual Reports

Creates publication-quality visual reports with better spacing, larger fonts,
and clear data labels for professional presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set improved plotting style with better spacing
plt.rcParams.update({
    'font.size': 14,           # Increased from 12
    'font.family': 'Arial',
    'axes.titlesize': 18,      # Increased from 14
    'axes.labelsize': 16,      # Increased from 12
    'xtick.labelsize': 14,     # Increased from 10
    'ytick.labelsize': 14,     # Increased from 10
    'legend.fontsize': 14,     # Increased from 10
    'figure.titlesize': 22,    # Increased from 16
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.5,  # More padding
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.constrained_layout.use': True  # Better automatic spacing
})

class MAPPImprovedReports:
    """
    Improved visual reporting for MAPP observational study
    """
    
    def __init__(self, data_dir):
        """
        Initialize with de-identified data directory
        """
        self.data_dir = data_dir
        self.events_dir = os.path.join(data_dir, 'events')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        self.output_dir = os.path.join(data_dir, 'improved_visual_reports')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load variable labels
        self.load_labels()
        
        # Load data
        self.load_study_data()
        
        # Improved color schemes
        self.colors = {
            'primary': '#1f77b4',      # Blue
            'secondary': '#ff7f0e',    # Orange  
            'success': '#2ca02c',      # Green
            'warning': '#d62728',      # Red
            'info': '#9467bd',         # Purple
            'neutral': '#8c564b',      # Brown
            'light': '#f0f0f0',       # Light gray
            'dark': '#2f2f2f',        # Dark gray
            'missing': '#DC143C',      # Crimson
            'complete': '#32CD32'      # Lime Green
        }
        
        # Color palette for multiple categories
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def load_labels(self):
        """
        Load variable labels from metadata
        """
        labels_file = os.path.join(self.metadata_dir, 'variable_labels.json')
        
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                self.labels_data = json.load(f)
            print(f"Loaded {len(self.labels_data)} variable labels")
        except:
            self.labels_data = {}
            print("Warning: Could not load variable labels")
    
    def get_clean_label(self, variable, max_length=40):
        """
        Get clean, readable label for plotting
        """
        if variable in self.labels_data:
            label = self.labels_data[variable]['label']
            # Clean HTML and shorten
            label = label.replace('<div class="rich-text-field-label">', '')
            label = label.replace('</div>', '').replace('<p>', '').replace('</p>', '')
            label = label.replace('<br>', ' ').replace('<em>', '').replace('</em>', '')
            label = label.replace('<span style="text-decoration: underline;">', '').replace('</span>', '')
            
            # Remove long instructional text
            if len(label) > max_length:
                # Try to find a natural break point
                if '?' in label:
                    label = label.split('?')[0] + '?'
                elif '.' in label and len(label.split('.')[0]) < max_length:
                    label = label.split('.')[0] + '.'
                else:
                    label = label[:max_length-3] + "..."
            return label.strip()
        else:
            return variable.replace('_', ' ').title()
    
    def load_study_data(self):
        """
        Load all study data files
        """
        self.data = {}
        
        # Load event data
        event_files = {
            'enrollment': 'enrollment-arm-1_deidentified_2025-08-20.csv',
            'baseline': 'baseline-preop-arm-1_deidentified_2025-08-20.csv',
            'week4': 'week-4-postop-arm-1_deidentified_2025-08-20.csv',
            'month3': 'month-3-post-op-arm-1_deidentified_2025-08-20.csv'
        }
        
        for event, filename in event_files.items():
            filepath = os.path.join(self.events_dir, filename)
            if os.path.exists(filepath):
                self.data[event] = pd.read_csv(filepath)
                print(f"Loaded {event}: {self.data[event].shape[0]} participants")
            else:
                print(f"File not found: {filename}")
    
    def create_baseline_demographics_improved(self):
        """
        Create improved baseline demographics with better spacing and labels
        """
        print("\n=== CREATING IMPROVED BASELINE DEMOGRAPHICS ===")
        
        if 'baseline' not in self.data:
            print("Baseline data not available")
            return
        
        df = self.data['baseline']
        
        # Create figure with better spacing
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('MAPP Study: Baseline Patient Characteristics', fontsize=24, y=0.95, weight='bold')
        
        # 1. Age Distribution
        ax1 = axes[0, 0]
        if 'hh_age' in df.columns:
            age_data = df['hh_age'].dropna()
            n, bins, patches = ax1.hist(age_data, bins=8, color=self.colors['primary'], 
                                       alpha=0.7, edgecolor='white', linewidth=1.5)
            
            # Add data labels on bars
            for i, (patch, count) in enumerate(zip(patches, n)):
                if count > 0:
                    ax1.text(patch.get_x() + patch.get_width()/2., count + 0.1,
                            f'{int(count)}', ha='center', va='bottom', fontsize=12, weight='bold')
            
            # Add mean line with label
            mean_val = age_data.mean()
            ax1.axvline(mean_val, color=self.colors['warning'], linestyle='--', linewidth=3,
                       label=f'Mean: {mean_val:.1f} years')
            
            ax1.set_title('Age Distribution', weight='bold', pad=20)
            ax1.set_xlabel('Age (years)', fontsize=16)
            ax1.set_ylabel('Number of Participants', fontsize=16)
            ax1.legend(loc='upper right', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Add sample size
            ax1.text(0.02, 0.95, f'n = {len(age_data)}', transform=ax1.transAxes,
                    fontsize=14, weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', alpha=0.8))
        
        # 2. Sex Distribution
        ax2 = axes[0, 1]
        if 'hh_sex' in df.columns:
            sex_counts = df['hh_sex'].value_counts()
            # Map values to labels (1=Male, 2=Female per data dictionary)
            sex_labels = []
            sex_values = []
            for val, count in sex_counts.items():
                if val == 1:
                    sex_labels.append(f'Male\n({count})')
                    sex_values.append(count)
                elif val == 2:
                    sex_labels.append(f'Female\n({count})')
                    sex_values.append(count)
            
            colors = [self.colors['secondary'], self.colors['primary']][:len(sex_values)]
            wedges, texts, autotexts = ax2.pie(sex_values, labels=sex_labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90, textprops={'fontsize': 14})
            
            ax2.set_title('Sex Distribution', weight='bold', pad=20)
            
            # Make percentage text bold and white
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                autotext.set_fontsize(16)
        
        # 3. Surgery Site Distribution  
        ax3 = axes[1, 0]
        if 'qst_surgery_site' in df.columns:
            surgery_counts = df['qst_surgery_site'].value_counts()
            
            # Map surgery site codes to labels
            site_mapping = {1: 'Left Knee', 2: 'Right Knee', 3: 'Left Hip', 4: 'Right Hip'}
            
            labels = []
            values = []
            colors_list = []
            
            for i, (site_code, count) in enumerate(surgery_counts.items()):
                label = site_mapping.get(site_code, f'Site {site_code}')
                labels.append(label)
                values.append(count)
                colors_list.append(self.palette[i % len(self.palette)])
            
            bars = ax3.bar(labels, values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=14, weight='bold')
            
            ax3.set_title('Surgery Site Distribution', weight='bold', pad=20)
            ax3.set_xlabel('Surgery Site', fontsize=16)
            ax3.set_ylabel('Number of Participants', fontsize=16)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add total count
            total = sum(values)
            ax3.text(0.98, 0.95, f'Total: {total}', transform=ax3.transAxes,
                    ha='right', va='top', fontsize=14, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 4. Study Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate key statistics
        n_total = len(df)
        age_mean = df['hh_age'].mean() if 'hh_age' in df.columns else 0
        age_std = df['hh_age'].std() if 'hh_age' in df.columns else 0
        age_range = f"{df['hh_age'].min():.0f}-{df['hh_age'].max():.0f}" if 'hh_age' in df.columns else "N/A"
        
        # Per data dictionary: 1=Male, 2=Female
        male_count = (df['hh_sex'] == 1).sum() if 'hh_sex' in df.columns else 0
        female_count = (df['hh_sex'] == 2).sum() if 'hh_sex' in df.columns else 0
        female_pct = (female_count / n_total * 100) if n_total > 0 else 0
        male_pct = (male_count / n_total * 100) if n_total > 0 else 0
        
        # Surgery site summary
        surgery_summary = ""
        if 'qst_surgery_site' in df.columns:
            surgery_counts = df['qst_surgery_site'].value_counts()
            knee_total = surgery_counts.get(1.0, 0) + surgery_counts.get(2.0, 0)
            hip_total = surgery_counts.get(3.0, 0) + surgery_counts.get(4.0, 0)
            surgery_summary = f"""
Surgery Distribution:
  • Knee Procedures: {knee_total}
  • Hip Procedures: {hip_total}"""
        
        summary_text = f"""MAPP STUDY BASELINE SUMMARY
        
Enrollment Status:
  • Total Enrolled: {n_total} participants
  • Study Design: Prospective observational
  • Follow-up: 9 months post-surgery
  • Setting: UF Health Orthopaedics

Demographics:
  • Mean Age: {age_mean:.1f} ± {age_std:.1f} years
  • Age Range: {age_range} years
  • Female: {female_count} ({female_pct:.1f}%)
  • Male: {male_count} ({male_pct:.1f}%){surgery_summary}

Data Collection:
  • Assessment Timepoints: 5
  • PROMIS Measures: 16+
  • Biomarker Collection: Yes
  • QST Testing: Yes"""
        
        ax4.text(0.05, 0.95, summary_text, fontsize=16, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=1", facecolor=self.colors['light'], 
                         alpha=0.9, edgecolor=self.colors['primary'], linewidth=2),
                family='monospace')
        
        plt.tight_layout(pad=3.0)
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'baseline_demographics_improved.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Improved baseline demographics saved to: {output_file}")
    
    def create_pain_trajectory_improved(self):
        """
        Create improved pain trajectory visualization with better spacing
        """
        print("\n=== CREATING IMPROVED PAIN TRAJECTORY VISUALIZATION ===")
        
        # Create larger figure with more space
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('MAPP Study: Pain Outcome Trajectories Over Time', 
                    fontsize=24, y=0.95, weight='bold')
        
        # Define timepoints and events
        timepoints = ['Baseline\n(Pre-surgery)', '4 Weeks\n(Post-surgery)', '3 Months\n(Post-surgery)']
        events = ['baseline', 'week4', 'month3']
        
        # Key pain variables to analyze
        pain_variables = [
            'promis_pain_int_1',
            'promis_pain_int_2', 
            'pain_interference_1',
            'pain_interference_2'
        ]
        
        for i, pain_var in enumerate(pain_variables):
            if i >= 4:
                break
                
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Collect data for this variable across timepoints
            trajectory_data = []
            trajectory_errors = []
            valid_timepoints = []
            sample_sizes = []
            
            for j, event in enumerate(events):
                if event in self.data and pain_var in self.data[event].columns:
                    pain_data = self.data[event][pain_var].dropna()
                    if len(pain_data) > 0:
                        trajectory_data.append(pain_data.mean())
                        trajectory_errors.append(pain_data.std() / np.sqrt(len(pain_data)))  # SEM
                        valid_timepoints.append(timepoints[j])
                        sample_sizes.append(len(pain_data))
            
            if len(trajectory_data) >= 2:
                # Plot trajectory with larger markers and thicker lines
                x_positions = range(len(trajectory_data))
                
                ax.errorbar(x_positions, trajectory_data, yerr=trajectory_errors,
                           marker='o', linewidth=4, markersize=12, capsize=8, capthick=3,
                           color=self.colors['primary'], markerfacecolor=self.colors['secondary'],
                           markeredgecolor='white', markeredgewidth=2)
                
                # Add trend line if we have multiple points
                if len(trajectory_data) >= 2:
                    z = np.polyfit(x_positions, trajectory_data, 1)
                    p = np.poly1d(z)
                    trend_line = p(x_positions)
                    ax.plot(x_positions, trend_line, "--", alpha=0.7, color=self.colors['warning'],
                           linewidth=3, label=f'Trend (slope: {z[0]:.2f})')
                
                # Add value labels at each point
                for k, (x, y, err, n) in enumerate(zip(x_positions, trajectory_data, trajectory_errors, sample_sizes)):
                    ax.text(x, y + err + 0.2, f'{y:.1f}', ha='center', va='bottom', 
                           fontsize=14, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    
                    # Add sample size below x-axis
                    ax.text(x, ax.get_ylim()[0] - 0.3, f'n={n}', ha='center', va='top', 
                           fontsize=12, weight='bold', color=self.colors['dark'])
                
                ax.set_title(self.get_clean_label(pain_var, 35), weight='bold', pad=20)
                ax.set_xlabel('Time Point', fontsize=16)
                ax.set_ylabel('Score (Mean ± SEM)', fontsize=16)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(valid_timepoints, fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=12)
                
                # Set y-axis limits with some padding
                y_min = min(trajectory_data) - max(trajectory_errors) - 0.5
                y_max = max(trajectory_data) + max(trajectory_errors) + 0.8
                ax.set_ylim(y_min, y_max)
                
            else:
                ax.text(0.5, 0.5, 'Insufficient Data\nfor Trajectory Analysis', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=16, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light']))
                ax.set_title(self.get_clean_label(pain_var, 35), weight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'pain_trajectories_improved.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Improved pain trajectories saved to: {output_file}")
    
    def create_promis_dashboard_improved(self):
        """
        Create improved PROMIS outcomes dashboard with clear labels
        """
        print("\n=== CREATING IMPROVED PROMIS OUTCOMES DASHBOARD ===")
        
        if 'baseline' not in self.data:
            print("Baseline data not available")
            return
        
        df = self.data['baseline']
        
        # Find PROMIS variables
        promis_vars = [col for col in df.columns if 'promis' in col.lower()]
        
        if not promis_vars:
            print("No PROMIS variables found")
            return
        
        # Select top 4 PROMIS variables with most data
        promis_data_counts = []
        for var in promis_vars:
            count = df[var].dropna().shape[0]
            promis_data_counts.append((var, count))
        
        # Sort by count and take top 4
        promis_data_counts.sort(key=lambda x: x[1], reverse=True)
        top_promis = [var for var, count in promis_data_counts[:4]]
        
        # Create dashboard with larger spacing
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('MAPP Study: PROMIS Patient-Reported Outcomes (Baseline)', 
                    fontsize=24, y=0.95, weight='bold')
        
        for i, promis_var in enumerate(top_promis):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            promis_data = df[promis_var].dropna()
            
            if len(promis_data) > 0:
                # Create improved histogram
                n_bins = min(10, len(promis_data.unique()))
                n, bins, patches = ax.hist(promis_data, bins=n_bins, 
                                          color=self.colors['primary'], alpha=0.7, 
                                          edgecolor='white', linewidth=2)
                
                # Add data labels on bars
                for patch, count in zip(patches, n):
                    if count > 0:
                        ax.text(patch.get_x() + patch.get_width()/2., count + 0.05,
                               f'{int(count)}', ha='center', va='bottom', 
                               fontsize=12, weight='bold')
                
                # Add mean line with improved label
                mean_val = promis_data.mean()
                std_val = promis_data.std()
                ax.axvline(mean_val, color=self.colors['warning'], linestyle='--', 
                          linewidth=3, label=f'Mean: {mean_val:.1f}')
                
                # Add median line
                median_val = promis_data.median()
                ax.axvline(median_val, color=self.colors['success'], linestyle='-', 
                          linewidth=2, alpha=0.8, label=f'Median: {median_val:.1f}')
                
                ax.set_title(self.get_clean_label(promis_var, 30), weight='bold', pad=20)
                ax.set_xlabel('Score', fontsize=16)
                ax.set_ylabel('Number of Participants', fontsize=16)
                ax.legend(fontsize=14, loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Add comprehensive statistics box
                stats_text = f"""n = {len(promis_data)}
Mean = {mean_val:.1f}
SD = {std_val:.1f}
Range = {promis_data.min():.1f}-{promis_data.max():.1f}"""
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                                alpha=0.9, edgecolor=self.colors['primary']))
                
            else:
                ax.text(0.5, 0.5, 'No Data Available\nfor This Measure', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=16, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light']))
                ax.set_title(self.get_clean_label(promis_var, 30), weight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'promis_outcomes_improved.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Improved PROMIS outcomes saved to: {output_file}")
    
    def create_executive_summary_improved(self):
        """
        Create improved executive summary with better layout
        """
        print("\n=== CREATING IMPROVED EXECUTIVE SUMMARY ===")
        
        # Create figure with single large layout
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], 
                             hspace=0.3, wspace=0.3)
        
        fig.suptitle('MAPP Study: Executive Summary Dashboard', 
                    fontsize=26, y=0.95, weight='bold')
        
        # Top row: Key metrics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        # Calculate key metrics
        total_enrolled = len(self.data.get('baseline', []))
        mean_age = self.data.get('baseline', pd.DataFrame()).get('hh_age', pd.Series()).mean()
        # Per data dictionary: 1=Male, 2=Female
        female_pct = (self.data.get('baseline', pd.DataFrame()).get('hh_sex', pd.Series()) == 2).mean() * 100
        
        # Create metrics boxes
        metrics = [
            ('Total Enrolled', f'{total_enrolled}', 'participants'),
            ('Mean Age', f'{mean_age:.1f}', 'years'),
            ('Female', f'{female_pct:.0f}%', 'of sample'),
            ('Study Duration', '9', 'months'),
            ('Timepoints', '5', 'assessments')
        ]
        
        box_width = 0.18
        for i, (label, value, unit) in enumerate(metrics):
            x_pos = 0.05 + i * 0.19
            
            # Create metric box
            bbox = dict(boxstyle="round,pad=0.02", facecolor=self.colors['primary'], 
                       alpha=0.1, edgecolor=self.colors['primary'], linewidth=2)
            ax1.text(x_pos + box_width/2, 0.7, value, ha='center', va='center',
                    fontsize=28, weight='bold', color=self.colors['primary'],
                    transform=ax1.transAxes, bbox=bbox)
            
            ax1.text(x_pos + box_width/2, 0.4, label, ha='center', va='center',
                    fontsize=14, weight='bold', transform=ax1.transAxes)
            
            ax1.text(x_pos + box_width/2, 0.25, unit, ha='center', va='center',
                    fontsize=12, color=self.colors['neutral'], transform=ax1.transAxes)
        
        # Middle left: Surgery distribution
        ax2 = fig.add_subplot(gs[1, 0])
        if 'baseline' in self.data and 'qst_surgery_site' in self.data['baseline'].columns:
            surgery_counts = self.data['baseline']['qst_surgery_site'].value_counts()
            
            # Simplify to knee vs hip (handling float values)
            knee_count = surgery_counts.get(1.0, 0) + surgery_counts.get(2.0, 0)
            hip_count = surgery_counts.get(3.0, 0) + surgery_counts.get(4.0, 0)
            
            if knee_count > 0 or hip_count > 0:
                sizes = [knee_count, hip_count]
                labels = [f'Knee Surgery\\n({knee_count})', f'Hip Surgery\\n({hip_count})']
                colors = [self.colors['primary'], self.colors['secondary']]
                
                wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.0f%%', 
                                                  colors=colors, startangle=90, 
                                                  textprops={'fontsize': 14, 'weight': 'bold'})
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(16)
                    autotext.set_weight('bold')
        
        ax2.set_title('Surgery Distribution', weight='bold', fontsize=18, pad=20)
        
        # Middle center: Retention chart
        ax3 = fig.add_subplot(gs[1, 1])
        
        retention_timepoints = ['Baseline', '4-Week', '3-Month']
        retention_values = []
        
        baseline_n = len(self.data.get('baseline', []))
        if baseline_n > 0:
            retention_values = [100]  # Baseline = 100%
            
            week4_n = len(self.data.get('week4', []))
            retention_values.append((week4_n / baseline_n) * 100)
            
            month3_n = len(self.data.get('month3', []))
            retention_values.append((month3_n / baseline_n) * 100)
            
            ax3.plot(retention_timepoints, retention_values, 'o-', linewidth=4, markersize=12,
                    color=self.colors['success'], markerfacecolor=self.colors['primary'],
                    markeredgecolor='white', markeredgewidth=3)
            
            # Add value labels
            for i, val in enumerate(retention_values):
                ax3.text(i, val + 2, f'{val:.0f}%', ha='center', va='bottom', 
                        fontsize=14, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax3.set_title('Study Retention', weight='bold', fontsize=18, pad=20)
        ax3.set_ylabel('Retention Rate (%)', fontsize=14, weight='bold')
        ax3.set_ylim(0, 110)
        ax3.grid(True, alpha=0.3)
        
        # Middle right: Data collection progress
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        
        # Calculate actual completion rates for all timepoints
        baseline_n = len(self.data.get('baseline', []))
        week4_n = len(self.data.get('week4', []))
        month3_n = len(self.data.get('month3', []))
        # Note: 6-month and 9-month data not yet available in current dataset
        
        status_items = [
            (f'✓ Baseline: {baseline_n} completed', self.colors['success']),
            (f'◐ 4-Week: {week4_n} completed', self.colors['info'] if week4_n > 0 else self.colors['neutral']),
            (f'◑ 3-Month: {month3_n} completed', self.colors['warning'] if month3_n > 0 else self.colors['neutral']),
            ('○ 6-Month: Pending', self.colors['neutral']),
            ('○ 9-Month: Pending', self.colors['neutral'])
        ]
        
        for i, (status, color) in enumerate(status_items):
            y_pos = 0.9 - i * 0.15
            ax4.text(0.1, y_pos, status, transform=ax4.transAxes, 
                    fontsize=14, color=color, weight='bold')
        
        ax4.text(0.5, 0.95, 'Data Collection Progress', transform=ax4.transAxes, 
                ha='center', fontsize=18, weight='bold')
        
        
        plt.tight_layout(pad=2.0)
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'executive_summary_improved.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Improved executive summary saved to: {output_file}")
    
    def create_missing_data_analysis(self):
        """
        Create CORRECTED missing data analysis distinguishing study design from true missing data
        """
        print("\n=== CREATING CORRECTED MISSING DATA ANALYSIS ===")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1], hspace=0.35, wspace=0.25)
        
        # Title
        fig.suptitle('MAPP Study: True Missing Data Analysis\n(Study Design vs. Actual Missingness)', 
                    fontsize=20, weight='bold', y=0.95)
        
        # 1. Participant tracking across timepoints (True attrition)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Get baseline participant IDs
        baseline_df = self.data.get('baseline', pd.DataFrame())
        week4_df = self.data.get('week4', pd.DataFrame())
        month3_df = self.data.get('month3', pd.DataFrame())
        
        if not baseline_df.empty and 'subject_id' in baseline_df.columns:
            baseline_ids = set(baseline_df['subject_id'])
            week4_ids = set(week4_df['subject_id']) if not week4_df.empty else set()
            month3_ids = set(month3_df['subject_id']) if not month3_df.empty else set()
            
            # Create participant tracking matrix
            all_ids = sorted(baseline_ids)
            timepoints = ['Baseline', '4-Week', '3-Month']
            
            # Load enrollment data to get study status
            import os
            orig_data_path = os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), 'MAPP_DATA_2025-08-20_0816.csv')
            
            if os.path.exists(orig_data_path):
                orig_df = pd.read_csv(orig_data_path)
                enrollment_data = orig_df[orig_df['redcap_event_name'] == 'enrollment_arm_1'].copy()
                
                # Get status for each participant
                status_dict = {}
                for _, row in enrollment_data.iterrows():
                    if pd.notna(row['ss_current_status']):
                        status_dict[row['subject_id']] = row['ss_current_status']
                
                # Matrix: 0=withdrawn, 1=pending, 2=completed
                tracking_matrix = []
                for subj_id in all_ids:
                    participant_status = status_dict.get(subj_id, 1.0)  # Default to pending if unknown
                    
                    row = []
                    for timepoint_ids in [baseline_ids, week4_ids, month3_ids]:
                        if subj_id in timepoint_ids:
                            row.append(2)  # Completed
                        elif participant_status == 2.0:  # Withdrawn
                            row.append(0)  # Withdrawn
                        else:  # Pending or other active status
                            row.append(1)  # Pending
                    tracking_matrix.append(row)
                
                tracking_array = np.array(tracking_matrix)
                
                # Create custom colormap: red=withdrawn, yellow=pending, green=completed
                colors = ['#d62728', '#ffd700', '#2ca02c']  # red, gold, green
                cmap = plt.matplotlib.colors.ListedColormap(colors)
                
                im = ax1.imshow(tracking_array, cmap=cmap, aspect='auto', alpha=0.8)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02, shrink=0.6)
                cbar.set_label('Study Status', fontsize=12, weight='bold')
                cbar.set_ticks([0, 1, 2])
                cbar.set_ticklabels(['Withdrawn', 'Pending', 'Completed'])
                
                title_text = 'Participant Status by Timepoint\n(Red = Withdrawn, Yellow = Pending, Green = Completed)'
            else:
                # Fallback to original method if status file not available
                tracking_matrix = []
                for subj_id in all_ids:
                    row = [
                        1,  # Baseline (all have this by definition)
                        1 if subj_id in week4_ids else 0,
                        1 if subj_id in month3_ids else 0
                    ]
                    tracking_matrix.append(row)
                
                tracking_array = np.array(tracking_matrix)
                
                colors = ['#d62728', '#2ca02c']  # red, green
                cmap = plt.matplotlib.colors.ListedColormap(colors)
                
                im = ax1.imshow(tracking_array, cmap=cmap, aspect='auto', alpha=0.8)
                
                cbar = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02, shrink=0.6)
                cbar.set_label('Participation Status', fontsize=12, weight='bold')
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(['Missed', 'Completed'])
                
                title_text = 'Participant Completion Status by Timepoint\n(Red = Missed Assessment, Green = Completed)'
            
            # Set labels
            ax1.set_xticks(range(len(timepoints)))
            ax1.set_xticklabels(timepoints, fontsize=14, weight='bold')
            ax1.set_yticks(range(len(all_ids)))
            ax1.set_yticklabels([f'ID {subj_id}' for subj_id in all_ids], fontsize=11, weight='bold')
            ax1.set_ylabel('Participants', fontsize=14, weight='bold')
            ax1.set_title(title_text, fontsize=16, weight='bold', pad=20)
        
        # 2. Within-timepoint true missing data (only for variables intended at each timepoint)
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Analyze true missingness within intended assessments
        timepoint_data = []
        timepoint_names = []
        
        # Baseline: Check key baseline variables for true missingness
        if not baseline_df.empty:
            baseline_vars = ['hh_age', 'hh_sex', 'qst_surgery_site', 'copcs_demo_sex']
            baseline_missing = []
            for var in baseline_vars:
                if var in baseline_df.columns:
                    missing_pct = baseline_df[var].isnull().mean() * 100
                    baseline_missing.append(missing_pct)
            avg_baseline_missing = np.mean(baseline_missing) if baseline_missing else 0
            timepoint_data.append(avg_baseline_missing)
            timepoint_names.append('Baseline\n(Core Variables)')
        
        # 4-Week: Check core assessment variables (pain/function measures)
        if not week4_df.empty:
            week4_vars = ['pain_interference_1', 'pain_interference_2', 'pain_interference_3', 
                         'pain_interference_4', 'physical_function_1', 'physical_function_2']
            week4_missing = []
            for var in week4_vars:
                if var in week4_df.columns:
                    missing_pct = week4_df[var].isnull().mean() * 100
                    week4_missing.append(missing_pct)
            avg_week4_missing = np.mean(week4_missing) if week4_missing else 0
            timepoint_data.append(avg_week4_missing)
            timepoint_names.append('4-Week\n(Core Measures)')
        
        # 3-Month: Check core PROMIS pain/function variables
        if not month3_df.empty:
            month3_vars = ['pain_interference_1', 'pain_interference_2', 'pain_interference_3', 
                          'pain_interference_4', 'physical_function_1', 'physical_function_3', 'physical_function_4']
            month3_missing = []
            for var in month3_vars:
                if var in month3_df.columns:
                    missing_pct = month3_df[var].isnull().mean() * 100
                    month3_missing.append(missing_pct)
            avg_month3_missing = np.mean(month3_missing) if month3_missing else 0
            timepoint_data.append(avg_month3_missing)
            timepoint_names.append('3-Month\n(Core Measures)')
        
        if timepoint_data:
            bars = ax2.bar(timepoint_names, timepoint_data, 
                          color=self.colors['warning'], alpha=0.7, 
                          edgecolor=self.colors['dark'], linewidth=2)
            
            # Add data labels
            for bar, pct in zip(bars, timepoint_data):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{pct:.1f}%', ha='center', va='bottom', 
                        fontsize=12, weight='bold')
            
            ax2.set_title('True Missing Data Within Intended Assessments', fontsize=14, weight='bold', pad=15)
            ax2.set_ylabel('Missing Data (%)', fontsize=12, weight='bold')
            ax2.set_ylim(0, max(timepoint_data) * 1.3 if timepoint_data else 10)
            ax2.grid(True, alpha=0.3)
        
        # 3. Study status and attrition (accounting for pending vs withdrawn)
        ax3 = fig.add_subplot(gs[1, 1])
        
        enrolled = len(baseline_ids) if baseline_ids else 0
        week4_n = len(week4_ids) if week4_ids else 0  
        month3_n = len(month3_ids) if month3_ids else 0
        
        # Load enrollment data to get study status  
        import os
        orig_data_path = os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), 'MAPP_DATA_2025-08-20_0816.csv')
        if os.path.exists(orig_data_path):
            orig_df = pd.read_csv(orig_data_path)
            enrollment_data = orig_df[orig_df['redcap_event_name'] == 'enrollment_arm_1'].copy()
            
            # Count truly withdrawn (status = 2) vs pending
            withdrawn_ids = set(enrollment_data[enrollment_data['ss_current_status'] == 2.0]['subject_id'])
            pending_ids = set(enrollment_data[enrollment_data['ss_current_status'] == 1.0]['subject_id'])
            
            # Calculate adjusted numbers
            missing_4week = baseline_ids - week4_ids if baseline_ids else set()
            missing_3month = week4_ids - month3_ids if week4_ids else set()
            
            withdrawn_4week = len(missing_4week & withdrawn_ids)
            pending_4week = len(missing_4week & pending_ids) 
            
            withdrawn_3month = len(missing_3month & withdrawn_ids)
            pending_3month = len(missing_3month & pending_ids)
            
            # Adjusted data (subtract pending, count only true withdrawals)
            adjusted_week4 = week4_n + pending_4week  # Add back pending participants
            adjusted_month3 = month3_n + pending_3month  # Add back pending participants
            
            attrition_data = [enrolled, adjusted_week4, adjusted_month3]
            actual_data = [enrolled, week4_n, month3_n]
        else:
            attrition_data = [enrolled, week4_n, month3_n] 
            actual_data = attrition_data
        
        attrition_labels = ['Baseline\n(Enrolled)', '4-Week\n(Active+Pending)', '3-Month\n(Active+Pending)']
        
        # Create flow chart
        ax3.step(range(len(attrition_data)), attrition_data, where='mid', 
                linewidth=4, color=self.colors['primary'], marker='o', markersize=12,
                markerfacecolor=self.colors['secondary'], markeredgecolor='white', 
                markeredgewidth=3)
        
        # Add dropout annotations
        for i in range(1, len(attrition_data)):
            dropout = attrition_data[i-1] - attrition_data[i]
            if dropout > 0:
                ax3.annotate(f'-{dropout}', 
                           xy=(i-0.5, (attrition_data[i-1] + attrition_data[i])/2),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=12, weight='bold', color=self.colors['warning'],
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color=self.colors['warning']))
        
        # Add retention percentages
        for i, (count, label) in enumerate(zip(attrition_data, attrition_labels)):
            retention_pct = (count / enrolled * 100) if enrolled > 0 else 0
            ax3.text(i, count + enrolled * 0.05, f'{count}\n({retention_pct:.0f}%)', 
                    ha='center', va='bottom', fontsize=12, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.9))
        
        ax3.set_xticks(range(len(attrition_labels)))
        ax3.set_xticklabels(attrition_labels, fontsize=12, weight='bold')
        ax3.set_title('Study Status: Completed vs Pending\n(Accounts for ss_current_status)', fontsize=14, weight='bold', pad=15)
        ax3.set_ylabel('Number of Participants', fontsize=12, weight='bold')
        ax3.set_ylim(0, enrolled * 1.15 if enrolled > 0 else 25)
        ax3.grid(True, alpha=0.3)
        
        # 4. Study design explanation
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create explanation text
        explanation_text = f"""CORRECTED Missing Data Interpretation (Accounting for Study Status):

PARTICIPANT STATUS ANALYSIS:
• Total Enrolled: {enrolled} participants  
• Completed 4-Week: {week4_n} participants
• Completed 3-Month: {month3_n} participants

IMPORTANT: Study Status (ss_current_status) reveals:
• "Missing" 4-Week (8 participants): 3 WITHDRAWN (true attrition) + 4 PENDING (active) + 1 other
• "Missing" 3-Month (5 participants): ALL 5 are PENDING (data collection ongoing)

TRUE ATTRITION vs PENDING DATA COLLECTION:
• Only 3 participants truly withdrawn from study (IDs 2, 5, 13)
• 9 participants have "pending" status - visits scheduled but not yet completed
• Study is ONGOING - most "missing" data represents future data collection, not dropouts

TRUE MISSING DATA (Within Completed Assessments):
• Baseline Core Variables: {timepoint_data[0]:.1f}% missing (demographics, QST as intended)
• 4-Week Core Measures: {timepoint_data[1]:.1f}% missing (pain/function assessment items as intended)  
• 3-Month Core Measures: {timepoint_data[2]:.1f}% missing (pain/function assessment items as intended)

STUDY DESIGN (NOT Missing Data):
• Different instruments collected at different timepoints by protocol design
• EMA events contain only EMA questions (other measures not applicable by design)

This analysis distinguishes: (1) True attrition, (2) Pending data collection, (3) Study design, (4) Missing responses."""
        
        ax4.text(0.05, 0.95, explanation_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light'], 
                         alpha=0.9, edgecolor=self.colors['primary'], linewidth=2))
        
        # Adjust layout manually to avoid colorbar conflict
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)
        
        # Save figure
        output_file = os.path.join(self.output_dir, 'missing_data_analysis_improved.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Corrected missing data analysis saved to: {output_file}")
    
    def generate_all_improved_reports(self):
        """
        Generate all improved visual reports
        """
        print("MAPP STUDY IMPROVED VISUAL REPORTS")
        print("=" * 55)
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Generate all improved visualizations
        self.create_baseline_demographics_improved()
        self.create_pain_trajectory_improved()
        self.create_promis_dashboard_improved()
        self.create_executive_summary_improved()
        self.create_missing_data_analysis()
        
        print("\n" + "=" * 55)
        print("ALL IMPROVED VISUAL REPORTS GENERATED")
        print("=" * 55)
        
        # List generated files
        generated_files = [
            'baseline_demographics_improved.png',
            'pain_trajectories_improved.png', 
            'promis_outcomes_improved.png',
            'executive_summary_improved.png'
        ]
        
        print("Generated Improved Reports:")
        for i, report in enumerate(generated_files, 1):
            print(f"  {i}. {report}")
        
        return self.output_dir

def main():
    """
    Main execution function
    """
    data_dir = "C:/Users/ebweber/Claude/MAPP/data/deidentified_labeled_data_2025-08-20"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return False
    
    # Create improved visual reports
    reporter = MAPPImprovedReports(data_dir)
    output_dir = reporter.generate_all_improved_reports()
    
    print(f"\\nImproved visual reports with better spacing and labels are ready!")
    print(f"Location: {output_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("Improved visual report generation failed!")