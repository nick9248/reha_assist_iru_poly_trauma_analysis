import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('time_interval_analysis')
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(file_handler)

    return logger


def load_dataset(file_path, logger):
    """Load dataset with appropriate error handling."""
    try:
        df = pd.read_excel(file_path, dtype={"Schadennummer": str})
        df.columns = df.columns.str.strip()
        logger.info(f"Successfully loaded data from: {file_path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns only
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise


def analyze_time_intervals(df, output_folder, plots_folder, logger):
    """
    Analyze how problems identified in specific time intervals relate to healing duration.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset with all visits
    output_folder : str
        Path to save analysis results
    plots_folder : str
        Path to save plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Results of the time interval analysis
    """
    logger.info("Starting time interval analysis...")

    # Check if necessary columns exist
    required_columns = ['Schadennummer', 'Time_Interval', 'Days_Since_Accident']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return None

    # Ensure Time_Interval is numeric
    if df['Time_Interval'].dtype == 'object':
        logger.info("Converting Time_Interval to numeric")
        df['Time_Interval'] = pd.to_numeric(df['Time_Interval'], errors='coerce')

    # Create patient-level dataset with healing duration (max Days_Since_Accident for each patient)
    logger.info("Calculating healing duration as max Days_Since_Accident for each patient")
    patient_healing = df.groupby('Schadennummer')['Days_Since_Accident'].max().reset_index()
    patient_healing.columns = ['Schadennummer', 'Heilungsdauer']

    logger.info(f"Created healing duration data for {len(patient_healing)} patients")

    # Identify problem columns (body parts and other categories with Ja/Nein values)
    logger.info("Identifying problem columns...")
    problem_columns = []

    # Check column types and values
    for col in df.columns:
        try:
            # Exclude non-categorical columns
            if col in ['Schadennummer', 'Unfalldatum', 'Besuchsdatum', 'Days_Since_Accident',
                       'Time_Interval', 'Monat nach Unfall', 'Age_At_Accident']:
                continue

            # Check if column contains Ja/Nein values
            unique_values = set(df[col].dropna().astype(str).unique())
            if 'Ja' in unique_values:
                problem_columns.append(col)
                logger.info(f"Added column as problem indicator: {col}")
        except Exception as e:
            logger.warning(f"Error checking column {col}: {str(e)}")

    if not problem_columns:
        logger.error("No problem columns identified")
        return None

    logger.info(f"Identified {len(problem_columns)} problem columns")

    # Define time intervals for analysis
    time_intervals = sorted(df['Time_Interval'].dropna().unique())
    logger.info(f"Time intervals in data: {time_intervals}")

    # Create a dataset of problem counts by patient and time interval
    problem_counts = []

    for patient_id in df['Schadennummer'].unique():
        patient_data = df[df['Schadennummer'] == patient_id]

        # Get healing duration for this patient
        patient_healing_data = patient_healing[patient_healing['Schadennummer'] == patient_id]
        if len(patient_healing_data) == 0:
            logger.warning(f"No healing duration data found for patient {patient_id}")
            continue

        healing_duration = patient_healing_data['Heilungsdauer'].values[0]

        for interval in time_intervals:
            interval_data = patient_data[patient_data['Time_Interval'] == interval]

            if len(interval_data) > 0:
                # Count problems identified in this interval
                problem_count = 0
                for col in problem_columns:
                    if col in interval_data.columns:
                        # Convert to string to handle any data type
                        if 'Ja' in interval_data[col].astype(str).values:
                            problem_count += 1

                problem_counts.append({
                    'Schadennummer': patient_id,
                    'Time_Interval': interval,
                    'Problem_Count': problem_count,
                    'Heilungsdauer': healing_duration
                })

    if not problem_counts:
        logger.error("No problem counts generated")
        return None

    # Convert to DataFrame
    problem_counts_df = pd.DataFrame(problem_counts)
    logger.info(f"Created problem counts dataframe with {len(problem_counts_df)} rows")

    # Save the problem counts dataframe for inspection
    problem_counts_df.to_excel(os.path.join(output_folder, 'problem_counts_by_interval.xlsx'), index=False)

    # Calculate correlation between problem counts in each interval and healing duration
    correlation_results = {}
    significant_intervals = []

    for interval in time_intervals:
        interval_data = problem_counts_df[problem_counts_df['Time_Interval'] == interval]

        if len(interval_data) >= 5:  # Minimum sample size for correlation
            if interval_data['Problem_Count'].std() > 0:  # Check for variation in problem counts
                corr, p_value = stats.pearsonr(interval_data['Problem_Count'], interval_data['Heilungsdauer'])

                correlation_results[interval] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n': len(interval_data),
                    'significant': p_value < 0.05
                }

                if p_value < 0.05:
                    significant_intervals.append(interval)

                logger.info(
                    f"Interval {interval}: correlation={corr:.3f}, p-value={p_value:.4f}, n={len(interval_data)}")
            else:
                logger.warning(f"No variation in problem counts for interval {interval}")
        else:
            logger.warning(f"Insufficient data for interval {interval} (n={len(interval_data)})")

    if not correlation_results:
        logger.warning("No valid correlations could be calculated")
        return {
            'problem_counts_df': problem_counts_df,
            'correlation_results': {},
            'significant_intervals': []
        }

    # Create visualization of correlations by time interval
    if len(correlation_results) > 0:
        intervals = []
        correlations = []
        p_values = []
        sample_sizes = []

        for interval, results in correlation_results.items():
            intervals.append(f"Interval {int(interval)}")
            correlations.append(results['correlation'])
            p_values.append(results['p_value'])
            sample_sizes.append(results['n'])

        plt.figure(figsize=(12, 8))

        # Create correlation bar chart
        bars = plt.bar(intervals, correlations, color=[
            'green' if p < 0.05 else 'lightblue' for p in p_values
        ])

        # Add p-value and sample size annotations
        for i, (bar, p, n) in enumerate(zip(bars, p_values, sample_sizes)):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05 if height >= 0 else height - 0.1,
                f'p={p:.3f}\nn={n}',
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=10
            )

        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Correlation Between Problem Count and Healing Duration by Time Interval', fontsize=14)
        plt.xlabel('Time Interval (3-month periods)', fontsize=12)
        plt.ylabel('Correlation Coefficient (r)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(-1, 1)

        # Add significance explanation
        plt.text(
            0.02, 0.02,
            'Green bars indicate statistically significant correlations (p < 0.05)',
            transform=plt.gca().transAxes,
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, 'time_interval_correlations.png'), dpi=300)
        plt.close()

        logger.info(f"Created correlation visualization")

    # Create scatter plots for significant intervals
    for interval in significant_intervals:
        interval_data = problem_counts_df[problem_counts_df['Time_Interval'] == interval]

        plt.figure(figsize=(10, 6))
        plt.scatter(interval_data['Problem_Count'], interval_data['Heilungsdauer'], alpha=0.7)

        # Add regression line
        m, b = np.polyfit(interval_data['Problem_Count'], interval_data['Heilungsdauer'], 1)
        plt.plot(
            [interval_data['Problem_Count'].min(), interval_data['Problem_Count'].max()],
            [m * interval_data['Problem_Count'].min() + b, m * interval_data['Problem_Count'].max() + b],
            'r--'
        )

        # Add correlation info
        corr = correlation_results[interval]['correlation']
        p_value = correlation_results[interval]['p_value']

        plt.text(
            0.05, 0.95,
            f'Correlation: {corr:.3f}\np-value: {p_value:.4f}',
            transform=plt.gca().transAxes,
            fontsize=12,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        plt.title(f'Problems in Time Interval {int(interval)} vs. Healing Duration', fontsize=14)
        plt.xlabel('Number of Problems Identified', fontsize=12)
        plt.ylabel('Healing Duration (Days)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(plots_folder, f'interval_{int(interval)}_scatter.png'), dpi=300)
        plt.close()

        logger.info(f"Created scatter plot for interval {interval}")

    # Create problem count heatmap across time intervals if we have enough data
    if len(problem_counts_df) > 5:
        try:
            # Pivot the data to create a matrix of problem counts by patient and interval
            pivot_data = problem_counts_df.pivot_table(
                index='Schadennummer',
                columns='Time_Interval',
                values='Problem_Count',
                aggfunc='mean'
            ).fillna(0)

            if len(pivot_data) > 0:
                # Sort patients by healing duration
                healing_duration_sorted = patient_healing.sort_values('Heilungsdauer')
                sorted_indices = [idx for idx in healing_duration_sorted['Schadennummer'] if idx in pivot_data.index]

                if sorted_indices:
                    pivot_data = pivot_data.loc[sorted_indices]

                    plt.figure(figsize=(12, 10))
                    ax = sns.heatmap(
                        pivot_data,
                        cmap='viridis',
                        annot=True,
                        fmt='.0f',
                        linewidths=0.5,
                        cbar_kws={'label': 'Number of Problems'}
                    )

                    plt.title('Problem Counts Across Time Intervals by Patient (Sorted by Healing Duration)',
                              fontsize=14)
                    plt.xlabel('Time Interval (3-month periods)', fontsize=12)
                    plt.ylabel('Patient ID (sorted by healing duration)', fontsize=12)

                    # Add healing duration as text on the y-axis
                    ax2 = ax.twinx()
                    ax2.set_yticks(np.arange(0.5, len(sorted_indices), 1))
                    ax2.set_yticklabels([
                                            f"{healing_duration_sorted[healing_duration_sorted['Schadennummer'] == idx]['Heilungsdauer'].values[0]:.0f} days"
                                            for idx in sorted_indices])
                    ax2.set_ylabel('Healing Duration', fontsize=12)

                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_folder, 'problem_count_heatmap.png'), dpi=300)
                    plt.close()

                    logger.info(f"Created problem count heatmap")
        except Exception as e:
            logger.warning(f"Error creating heatmap: {str(e)}")

    # Save results to Excel
    results_df = pd.DataFrame([
        {
            'Time_Interval': interval,
            'Correlation': results['correlation'],
            'P_Value': results['p_value'],
            'Sample_Size': results['n'],
            'Significant': results['significant']
        }
        for interval, results in correlation_results.items()
    ])

    results_df.to_excel(os.path.join(output_folder, 'time_interval_correlations.xlsx'), index=False)

    return {
        'problem_counts_df': problem_counts_df,
        'correlation_results': correlation_results,
        'significant_intervals': significant_intervals
    }


def generate_summary_report(results, output_folder, logger):
    """Generate a summary report of the time interval analysis."""
    if not results or 'correlation_results' not in results:
        logger.error("No valid results to generate report")
        return None

    report_path = os.path.join(output_folder, 'time_interval_analysis_report.md')

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Time Interval Analysis of Polytrauma Recovery\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")

            # Introduction
            f.write("## Overview\n\n")
            f.write("This analysis examines how problems identified during specific time intervals ")
            f.write("in the recovery process relate to overall healing duration. ")
            f.write("Time intervals are defined as 3-month periods after the injury.\n\n")

            # Results summary
            f.write("## Key Findings\n\n")

            # Count number of significant intervals
            significant_intervals = results.get('significant_intervals', [])
            if significant_intervals:
                f.write(f"### Significant Time Intervals\n\n")
                f.write(f"The analysis identified **{len(significant_intervals)}** time ")
                f.write(f"interval{'s' if len(significant_intervals) != 1 else ''} ")
                f.write("with a statistically significant correlation between problem count ")
                f.write("and healing duration:\n\n")

                for interval in sorted(significant_intervals):
                    corr = results['correlation_results'][interval]['correlation']
                    p_value = results['correlation_results'][interval]['p_value']
                    direction = "positive" if corr > 0 else "negative"
                    f.write(f"- **Time Interval {int(interval)}** (months {(interval - 1) * 3} to {interval * 3}): ")
                    f.write(f"{direction} correlation (r = {corr:.3f}, p = {p_value:.4f})\n")

                f.write("\n")
            else:
                f.write("No time intervals showed a statistically significant correlation ")
                f.write("between problem count and healing duration. This suggests that ")
                f.write("the timing of problem identification may be less important than ")
                f.write("the type or severity of problems.\n\n")

            # Complete correlation table
            f.write("### Correlation by Time Interval\n\n")
            f.write("| Time Interval | Correlation | p-value | Sample Size | Significant |\n")
            f.write("|--------------|------------|---------|-------------|--------------|\n")

            for interval, results in sorted(results['correlation_results'].items()):
                corr = results['correlation']
                p_value = results['p_value']
                n = results['n']
                significant = "Yes" if results['significant'] else "No"

                f.write(f"| {int(interval)} | {corr:.3f} | {p_value:.4f} | {n} | {significant} |\n")

            f.write("\n")

            # Interpretation
            f.write("## Interpretation\n\n")

            if significant_intervals:
                for interval in sorted(significant_intervals):
                    corr = results['correlation_results'][interval]['correlation']
                    if corr > 0:
                        f.write(f"- **Time Interval {int(interval)}**: ")
                        f.write("A higher number of problems identified during this period is associated ")
                        f.write("with **longer healing duration**. This may indicate that problems ")
                        f.write("emerging during this timeframe are particularly challenging to resolve ")
                        f.write("or have a more substantial impact on the overall recovery process.\n\n")
                    else:
                        f.write(f"- **Time Interval {int(interval)}**: ")
                        f.write("A higher number of problems identified during this period is associated ")
                        f.write("with **shorter healing duration**. This counterintuitive finding might ")
                        f.write("suggest that identifying and addressing problems during this timeframe ")
                        f.write("leads to more effective interventions and better outcomes.\n\n")

            # Clinical implications
            f.write("## Clinical Implications\n\n")

            if significant_intervals:
                f.write("The findings suggest that:")

                positive_intervals = [int(i) for i in significant_intervals
                                      if results['correlation_results'][i]['correlation'] > 0]
                negative_intervals = [int(i) for i in significant_intervals
                                      if results['correlation_results'][i]['correlation'] < 0]

                if positive_intervals:
                    f.write("\n\n1. **Critical monitoring periods**: Time intervals ")
                    f.write(f"{', '.join(map(str, positive_intervals))} ")
                    f.write("appear to be critical periods where problems have a greater impact on ")
                    f.write("healing duration. Enhanced monitoring and intervention during these ")
                    f.write("periods may be particularly important.")

                if negative_intervals:
                    f.write("\n\n2. **Effective intervention windows**: Time intervals ")
                    f.write(f"{', '.join(map(str, negative_intervals))} ")
                    f.write("show a negative correlation, suggesting that effective problem ")
                    f.write("identification and intervention during these periods may lead to ")
                    f.write("better outcomes.")

                f.write("\n\n3. **Targeted resource allocation**: Resources could be strategically ")
                f.write("allocated to provide more intensive support during the identified ")
                f.write("critical periods.")
            else:
                f.write("The lack of significant correlations between problem timing and healing duration ")
                f.write("suggests that clinical resources should be allocated based on problem type and ")
                f.write("severity rather than focusing on specific time windows in the recovery process.")

            f.write("\n\n")

            # Limitations
            f.write("## Limitations\n\n")
            f.write("1. **Sample size**: The analysis is limited by the small sample size, particularly ")
            f.write("when broken down by time intervals.\n")
            f.write("2. **Problem definition**: The analysis counts problems without weighting them by ")
            f.write("severity or type, which may obscure important distinctions.\n")
            f.write("3. **Healing duration definition**: Healing duration is defined as time to last visit, ")
            f.write("which may not perfectly correspond to complete recovery.\n")
            f.write("4. **Causality**: Correlation does not imply causation; longer healing duration ")
            f.write("might lead to more problems being identified rather than vice versa.\n\n")

            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. **Problem type analysis**: Analyze how specific types of problems at different ")
            f.write("time intervals affect healing duration.\n")
            f.write("2. **Intervention analysis**: Examine how interventions implemented during specific ")
            f.write("time intervals affect outcomes.\n")
            f.write("3. **Time-based analysis**: Investigate how visit timing and frequency relate to ")
            f.write("healing duration.\n")
            f.write("4. **Professional reintegration analysis**: Analyze the relationship between temporal ")
            f.write("patterns and return-to-work outcomes.\n\n")

            f.write("*This report was automatically generated as part of the Polytrauma Analysis Project*")

        logger.info(f"Summary report created: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        return None


def time_interval_analysis():
    """Main function to perform time interval analysis."""
    # Load environment variables
    load_dotenv()

    # Get paths from environment variables
    base_dataset = os.getenv("DATASET", "data/output/step1/Polytrauma_Analysis_Processed.xlsx")
    output_folder = os.getenv("OUTPUT_FOLDER", "data/output")
    log_folder = os.getenv("LOG_FOLDER", "logs")
    graphs_folder = os.getenv("GRAPHS_FOLDER", "plots")

    # Create folder structure for temporal analysis
    temporal_output_folder = os.path.join(output_folder, "step4", "temporal_analysis", "time_interval_analysis")
    temporal_log_folder = os.path.join(log_folder, "step4")
    temporal_plots_folder = os.path.join(graphs_folder, "step4", "temporal_analysis", "time_interval_analysis")

    # Create necessary directories
    os.makedirs(temporal_output_folder, exist_ok=True)
    os.makedirs(temporal_log_folder, exist_ok=True)
    os.makedirs(temporal_plots_folder, exist_ok=True)

    # Setup logging
    logger = setup_logging(temporal_log_folder, "time_interval_analysis.log")
    logger.info("Starting time interval analysis...")

    try:
        # Load the dataset
        df = load_dataset(base_dataset, logger)

        # Perform time interval analysis
        results = analyze_time_intervals(df, temporal_output_folder, temporal_plots_folder, logger)

        # Generate summary report
        if results:
            report_path = generate_summary_report(results, temporal_output_folder, logger)
            logger.info(f"Analysis completed successfully. Report saved to {report_path}")
        else:
            logger.error("Analysis did not produce valid results")

        logger.info("Time interval analysis completed")

    except Exception as e:
        logger.error(f"Error in time interval analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    time_interval_analysis()