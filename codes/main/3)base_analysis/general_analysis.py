import pandas as pd
import os
import matplotlib.pyplot as plt
import logging
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from scipy.stats import norm, skew, kurtosis
import seaborn as sns
from pathlib import Path


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('general_analysis')
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
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise


def save_enhanced_plot(data, title, x_labels, y_label, output_path, logger, palette='viridis'):
    """
    Save an enhanced bar chart with value annotations, improved styling, and error bars if applicable.

    Parameters:
    -----------
    data : list or dict
        The data to plot. If dict, keys are used as x_labels.
    title : str
        Title of the chart
    x_labels : list
        Labels for the x-axis
    y_label : str
        Label for the y-axis
    output_path : str
        Path to save the plot
    logger : logging.Logger
        Logger instance
    palette : str
        Color palette to use
    """
    plt.figure(figsize=(12, 7))

    # Set the style
    sns.set_style("whitegrid")

    # Create the bars with a modern color palette
    if isinstance(data, dict):
        x_labels = list(data.keys())
        values = list(data.values())
        bars = plt.bar(x_labels, values, color=sns.color_palette(palette, len(data)), edgecolor="black", alpha=0.8)
    else:
        bars = plt.bar(x_labels, data, color=sns.color_palette(palette, len(data)), edgecolor="black", alpha=0.8)

    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black"
        )

    # Add titles and labels with improved styling
    plt.title(title, fontsize=18, fontweight="bold", pad=20)
    plt.xlabel(x_labels[0] if len(x_labels) == 1 else "Kategorien", fontsize=14, fontweight="bold")
    plt.ylabel(y_label, fontsize=14, fontweight="bold")

    # Add grid for better readability
    plt.grid(visible=True, alpha=0.3, linestyle="--")

    # Format the axes
    plt.tick_params(axis='both', which='major', labelsize=12)
    if len(x_labels) > 6:
        plt.xticks(rotation=45, ha='right')

    # Add a subtle background color
    plt.gca().set_facecolor("#f8f9fa")

    # Add a border around the plot
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add tight layout and save with high resolution
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Enhanced chart saved: {output_path}")


def create_enhanced_age_distribution(df, plot_folder, logger):
    """
    Create an enhanced age distribution visualization with statistical insights.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the age data
    plot_folder : str
        Path to save the plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Age statistics
    """
    try:
        # For age, need to use a birth date column
        birthdate_col = "Gebursdatum" if "Gebursdatum" in df.columns else None

        if not birthdate_col:
            logger.warning("No birth date column found. Skipping age distribution analysis.")
            return None

        # Ensure datetime format
        df[birthdate_col] = pd.to_datetime(df[birthdate_col], errors='coerce', dayfirst=True)

        # Calculate age
        current_year = datetime.now().year
        df["Age"] = current_year - df[birthdate_col].dt.year

        # Use only distinct cases
        unique_cases = df.drop_duplicates(subset="Schadennummer")
        ages = unique_cases["Age"].dropna()

        if len(ages) == 0:
            logger.warning("No valid age data found. Skipping age distribution analysis.")
            return None

        # Calculate statistics
        mean_age = ages.mean()
        median_age = ages.median()
        std_dev_age = ages.std()
        min_age = ages.min()
        max_age = ages.max()
        skewness = skew(ages)
        kurt = kurtosis(ages)

        logger.info(f"Age Statistics: Mean={mean_age:.1f}, Median={median_age:.1f}, "
                    f"StdDev={std_dev_age:.1f}, Min={min_age}, Max={max_age}, "
                    f"Skewness={skewness:.2f}, Kurtosis={kurt:.2f}")

        # Create a figure for multiple subplots with better formatting
        fig = plt.figure(figsize=(20, 15))

        # Define a colormap for consistent colors
        color_palette = sns.color_palette("viridis", 4)

        # 1. Enhanced Histogram with KDE
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        sns.histplot(ages, kde=True, bins=10, color=color_palette[0], edgecolor="black", alpha=0.7, ax=ax1)
        ax1.set_title("Altersverteilung (Histogramm mit KDE)", fontsize=16, fontweight="bold")
        ax1.set_xlabel("Alter (Jahre)", fontsize=14)
        ax1.set_ylabel("Häufigkeit", fontsize=14)

        # Add mean and median lines
        ax1.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mittelwert: {mean_age:.1f}')
        ax1.axvline(median_age, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_age:.1f}')
        ax1.legend(fontsize=12)

        # 2. Enhanced Boxplot
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        sns.boxplot(x=ages, ax=ax2, color=color_palette[1])
        ax2.set_title("Altersverteilung (Boxplot)", fontsize=16, fontweight="bold")
        ax2.set_xlabel("Alter (Jahre)", fontsize=14)

        # Add text annotations for key statistics
        stats_text = (f"Mittelwert: {mean_age:.1f}\nMedian: {median_age:.1f}\nStd.Abw: {std_dev_age:.1f}\n"
                      f"Min: {min_age}\nMax: {max_age}\nSchiefe: {skewness:.2f}\nKurtosis: {kurt:.2f}")
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 3. Decade Bar Chart (Enhanced)
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        df["Decade"] = (df["Age"] // 10) * 10
        decade_counts = df.groupby("Decade")["Schadennummer"].nunique()
        decade_counts.index = decade_counts.index.astype(str) + 'er'

        bars = ax3.bar(decade_counts.index, decade_counts.values, color=color_palette[2], edgecolor="black")
        ax3.set_title("Patientenverteilung nach Altersdekade", fontsize=16, fontweight="bold")
        ax3.set_xlabel("Altersdekade", fontsize=14)
        ax3.set_ylabel("Anzahl der Patienten", fontsize=14)

        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 4. Age Distribution with Normal Curve
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        sns.histplot(ages, bins=10, kde=False, color=color_palette[3], edgecolor="black", alpha=0.7, ax=ax4)

        # Add normal distribution curve
        x = np.linspace(min_age - 5, max_age + 5, 100)
        y = norm.pdf(x, mean_age, std_dev_age) * len(ages) * (max_age - min_age) / 10
        ax4.plot(x, y, 'r-', linewidth=2, label='Normalverteilung')

        ax4.set_title("Altersverteilung mit Normalverteilungskurve", fontsize=16, fontweight="bold")
        ax4.set_xlabel("Alter (Jahre)", fontsize=14)
        ax4.set_ylabel("Häufigkeit", fontsize=14)
        ax4.legend(fontsize=12)

        # Add normality test result
        from scipy import stats as scipy_stats
        stat, p_value = scipy_stats.normaltest(ages)
        normality_result = "Normalverteilt" if p_value > 0.05 else "Nicht normalverteilt"
        ax4.text(0.05, 0.95, f"Normalitätstest: {normality_result}\np-Wert: {p_value:.4f}",
                 transform=ax4.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add overall title and adjust layout
        plt.suptitle("Erweiterte Altersverteilungsanalyse", fontsize=24, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the combined figure
        output_path = os.path.join(plot_folder, "enhanced_age_distribution_combined.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Enhanced age distribution saved to: {output_path}")

        return {
            "mean_age": mean_age,
            "median_age": median_age,
            "std_dev_age": std_dev_age,
            "min_age": min_age,
            "max_age": max_age,
            "skewness": skewness,
            "kurtosis": kurt,
            "normality_p_value": p_value
        }

    except Exception as e:
        logger.error(f"Error in age distribution analysis: {str(e)}", exc_info=True)
        return None


def analyze_visit_patterns(df, plot_folder, logger):
    """
    Perform enhanced analysis of patient visit patterns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the visit data
    plot_folder : str
        Path to save the plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Visit pattern statistics
    """
    try:
        # Calculate visit frequency per patient
        visits_per_patient = df["Schadennummer"].value_counts()

        # Basic statistics
        min_visits = visits_per_patient.min()
        max_visits = visits_per_patient.max()
        avg_visits = visits_per_patient.mean()
        median_visits = visits_per_patient.median()
        std_dev_visits = visits_per_patient.std()

        # Create histogram of visit counts
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        # Enhanced histogram with KDE
        sns.histplot(visits_per_patient, kde=True, bins=min(15, max_visits - min_visits + 1),
                     color=sns.color_palette("viridis")[3], edgecolor="black")

        # Add vertical lines for mean and median
        plt.axvline(avg_visits, color='red', linestyle='--', linewidth=2, label=f'Mittelwert: {avg_visits:.2f}')
        plt.axvline(median_visits, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_visits:.1f}')

        # Add titles and labels
        plt.title("Verteilung der Besuche pro Patient", fontsize=18, fontweight="bold", pad=20)
        plt.xlabel("Anzahl der Besuche", fontsize=14, fontweight="bold")
        plt.ylabel("Anzahl der Patienten", fontsize=14, fontweight="bold")
        plt.grid(visible=True, alpha=0.3, linestyle="--")
        plt.legend(fontsize=12)

        # Add statistics text box
        stats_text = (f"Mittelwert: {avg_visits:.2f}\nMedian: {median_visits:.1f}\n"
                      f"Min: {min_visits}\nMax: {max_visits}\nStd.Abw: {std_dev_visits:.2f}")
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save the plot
        output_path = os.path.join(plot_folder, "visit_frequency_distribution.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visit frequency distribution saved to: {output_path}")

        # Create a visualization of basic visit statistics
        stats_data = [min_visits, median_visits, avg_visits, max_visits]
        stats_labels = ["Minimum", "Median", "Durchschnitt", "Maximum"]

        save_enhanced_plot(
            stats_data,
            "Patientenbesuchshäufigkeit Statistiken",
            stats_labels,
            "Anzahl der Besuche",
            os.path.join(plot_folder, "visit_frequency_stats.png"),
            logger
        )

        # Time analysis if relevant columns exist
        if "Unfalldatum" in df.columns and "Besuchsdatum" in df.columns:
            # Ensure datetime format
            df["Unfalldatum"] = pd.to_datetime(df["Unfalldatum"], errors='coerce', dayfirst=True)
            df["Besuchsdatum"] = pd.to_datetime(df["Besuchsdatum"], errors='coerce', dayfirst=True)

            # Calculate days after accident for each visit
            df["Tage_nach_Unfall"] = (df["Besuchsdatum"] - df["Unfalldatum"]).dt.days

            # First visit timing
            first_visits = df.groupby("Schadennummer")["Tage_nach_Unfall"].min()

            # Last visit timing
            last_visits = df.groupby("Schadennummer")["Tage_nach_Unfall"].max()

            # Actual healing duration (Heilungsdauer) is the time from accident to last visit
            healing_duration = last_visits

            # Create a comprehensive visualization of visit timing
            plt.figure(figsize=(15, 10))

            # 1. First subplot: First visit timing histogram
            plt.subplot(2, 2, 1)
            sns.histplot(first_visits, kde=True, bins=10, color=sns.color_palette("viridis")[0])
            plt.title("Zeitpunkt des ersten Besuchs", fontsize=16, fontweight="bold")
            plt.xlabel("Tage nach Unfall", fontsize=14)
            plt.ylabel("Anzahl der Patienten", fontsize=14)

            # Add statistics
            first_visit_stats = (f"Min: {first_visits.min()}\nMax: {first_visits.max()}\n"
                                 f"Mittelwert: {first_visits.mean():.1f}\nMedian: {first_visits.median():.1f}")
            plt.text(0.95, 0.95, first_visit_stats, transform=plt.gca().transAxes, fontsize=12,
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 2. Second subplot: Last visit timing histogram
            plt.subplot(2, 2, 2)
            sns.histplot(last_visits, kde=True, bins=10, color=sns.color_palette("viridis")[1])
            plt.title("Zeitpunkt des letzten Besuchs", fontsize=16, fontweight="bold")
            plt.xlabel("Tage nach Unfall", fontsize=14)
            plt.ylabel("Anzahl der Patienten", fontsize=14)

            # Add statistics
            last_visit_stats = (f"Min: {last_visits.min()}\nMax: {last_visits.max()}\n"
                                f"Mittelwert: {last_visits.mean():.1f}\nMedian: {last_visits.median():.1f}")
            plt.text(0.95, 0.95, last_visit_stats, transform=plt.gca().transAxes, fontsize=12,
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 3. Third subplot: Healing duration histogram (Zeit von Unfall bis letztem Besuch)
            plt.subplot(2, 2, 3)
            sns.histplot(healing_duration, kde=True, bins=10, color=sns.color_palette("viridis")[2])
            plt.title("Heilungsdauer", fontsize=16, fontweight="bold")
            plt.xlabel("Tage (Unfall bis letzter Besuch)", fontsize=14)
            plt.ylabel("Anzahl der Patienten", fontsize=14)

            # Add statistics
            healing_stats = (f"Min: {healing_duration.min()}\nMax: {healing_duration.max()}\n"
                             f"Mittelwert: {healing_duration.mean():.1f}\nMedian: {healing_duration.median():.1f}")
            plt.text(0.95, 0.95, healing_stats, transform=plt.gca().transAxes, fontsize=12,
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 4. Fourth subplot: Scatter plot of first vs. last visit
            plt.subplot(2, 2, 4)
            plt.scatter(first_visits, last_visits, c=healing_duration, cmap='viridis',
                        alpha=0.8, edgecolors='k', s=100)
            plt.colorbar(label='Heilungsdauer (Tage)')
            plt.title("Erster vs. Letzter Besuch", fontsize=16, fontweight="bold")
            plt.xlabel("Erster Besuch (Tage nach Unfall)", fontsize=14)
            plt.ylabel("Letzter Besuch (Tage nach Unfall)", fontsize=14)

            # Add regression line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(first_visits, last_visits)
            x = np.array([first_visits.min(), first_visits.max()])
            plt.plot(x, intercept + slope * x, 'r--', linewidth=2)

            # Add correlation info
            corr_info = f"Korrelation: {r_value:.2f}\np-Wert: {p_value:.4f}"
            plt.text(0.05, 0.95, corr_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Save the combined plot
            plt.suptitle("Erweiterte Analyse der Besuchszeitpunkte", fontsize=24, fontweight="bold", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            output_path = os.path.join(plot_folder, "visit_timing_analysis.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Enhanced visit timing analysis saved to: {output_path}")

            return {
                "visits_statistics": {
                    "min_visits": min_visits,
                    "max_visits": max_visits,
                    "avg_visits": avg_visits,
                    "median_visits": median_visits,
                    "std_dev_visits": std_dev_visits
                },
                "timing_statistics": {
                    "first_visit_min": first_visits.min(),
                    "first_visit_max": first_visits.max(),
                    "first_visit_mean": first_visits.mean(),
                    "first_visit_median": first_visits.median(),
                    "last_visit_min": last_visits.min(),
                    "last_visit_max": last_visits.max(),
                    "last_visit_mean": last_visits.mean(),
                    "last_visit_median": last_visits.median(),
                    "healing_duration_min": healing_duration.min(),
                    "healing_duration_max": healing_duration.max(),
                    "healing_duration_mean": healing_duration.mean(),
                    "healing_duration_median": healing_duration.median(),
                    "correlation_first_last": r_value,
                    "correlation_p_value": p_value
                }
            }

        else:
            logger.warning("Columns 'Unfalldatum' and/or 'Besuchsdatum' not found. Skipping visit timing analysis.")
            return {
                "visits_statistics": {
                    "min_visits": min_visits,
                    "max_visits": max_visits,
                    "avg_visits": avg_visits,
                    "median_visits": median_visits,
                    "std_dev_visits": std_dev_visits
                }
            }

    except Exception as e:
        logger.error(f"Error in visit pattern analysis: {str(e)}", exc_info=True)
        return None


def analyze_gender_distribution(df, plot_folder, logger):
    """
    Analyze and visualize the gender distribution of patients.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the gender data
    plot_folder : str
        Path to save the plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Gender distribution statistics
    """
    try:
        if "Geschlecht" not in df.columns:
            logger.warning("Column 'Geschlecht' not found. Skipping gender distribution analysis.")
            return None

        # Use only distinct cases
        unique_cases = df.drop_duplicates(subset="Schadennummer")

        # Count genders
        gender_counts = unique_cases["Geschlecht"].value_counts()

        # Calculate percentages
        gender_percentages = (gender_counts / gender_counts.sum() * 100).round(1)

        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                startangle=90, shadow=True, explode=[0.05] * len(gender_counts),
                colors=sns.color_palette("viridis", len(gender_counts)), textprops={'fontsize': 14})

        plt.title("Geschlechterverteilung der Patienten", fontsize=18, fontweight="bold")
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Save the plot
        output_path = os.path.join(plot_folder, "gender_distribution.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Gender distribution chart saved to: {output_path}")

        # Create bar chart for comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(gender_counts.index, gender_counts.values,
                       color=sns.color_palette("viridis", len(gender_counts)),
                       edgecolor="black", alpha=0.8)

        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height} ({height / sum(gender_counts.values) * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold"
            )

        plt.title("Geschlechterverteilung (Anzahl und Prozentsatz)", fontsize=18, fontweight="bold")
        plt.xlabel("Geschlecht", fontsize=14, fontweight="bold")
        plt.ylabel("Anzahl der Patienten", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, linestyle="--")

        # Save the bar chart
        output_path = os.path.join(plot_folder, "gender_distribution_bar.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Gender distribution bar chart saved to: {output_path}")

        return {
            "gender_counts": gender_counts.to_dict(),
            "gender_percentages": gender_percentages.to_dict()
        }

    except Exception as e:
        logger.error(f"Error in gender distribution analysis: {str(e)}", exc_info=True)
        return None


def export_statistics_to_excel(statistics, output_folder, logger):
    """
    Export all statistics to Excel for easy reference.

    Parameters:
    -----------
    statistics : dict
        Dictionary containing all statistics
    output_folder : str
        Path to save the Excel file
    logger : logging.Logger
        Logger instance
    """
    try:
        # Create a DataFrame from the flattened statistics dictionary
        flat_stats = {}

        for category, stats in statistics.items():
            if stats is None:
                continue

            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_stats[f"{category}_{key}_{subkey}"] = subvalue
                    else:
                        flat_stats[f"{category}_{key}"] = value
            else:
                flat_stats[category] = stats

        # Convert to DataFrame and transpose for better readability
        stats_df = pd.DataFrame([flat_stats]).T.reset_index()
        stats_df.columns = ["Statistik", "Wert"]

        # Add a "Category" column
        stats_df["Kategorie"] = stats_df["Statistik"].apply(lambda x: x.split("_")[0] if "_" in x else "Allgemein")

        # Reorder columns
        stats_df = stats_df[["Kategorie", "Statistik", "Wert"]]

        # Save to Excel
        output_path = os.path.join(output_folder, "general_analysis_statistics.xlsx")
        stats_df.to_excel(output_path, index=False)

        logger.info(f"Statistics exported to Excel: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error exporting statistics to Excel: {str(e)}", exc_info=True)
        return None


def general_analysis():
    """Main function to perform the optimized general analysis."""
    # Load environment variables
    load_dotenv()

    # Get paths from environment variables
    base_input_excel = os.getenv("DATASET")
    output_folder = os.getenv("OUTPUT_FOLDER", "data/output")
    log_folder = os.getenv("LOG_FOLDER", "logs")
    graphs_folder = os.getenv("GRAPHS_FOLDER", "plots")

    # Create step3 folder structure
    step3_output_folder = os.path.join(output_folder, "step3")
    step3_log_folder = os.path.join(log_folder, "step3")
    step3_plots_folder = os.path.join(graphs_folder, "step3", "general_analysis")

    # Create necessary directories
    os.makedirs(step3_output_folder, exist_ok=True)
    os.makedirs(step3_log_folder, exist_ok=True)
    os.makedirs(step3_plots_folder, exist_ok=True)

    # Setup logging
    logger = setup_logging(step3_log_folder, "general_analysis.log")
    logger.info("Starting enhanced general analysis...")

    try:
        # Load the dataset
        df = load_dataset(base_input_excel, logger)

        # 1. Count distinct cases
        distinct_cases = df["Schadennummer"].nunique()
        logger.info(f"Number of distinct cases: {distinct_cases}")

        # Create basic statistics dictionary
        statistics = {
            "distinct_cases": distinct_cases,
            "total_records": len(df),
            "columns_count": len(df.columns)
        }

        # Create an enhanced bar chart for distinct cases
        save_enhanced_plot(
            [distinct_cases],
            "Anzahl der eindeutigen Patientenfälle",
            ["Eindeutige Fälle"],
            "Anzahl",
            os.path.join(step3_plots_folder, "distinct_cases.png"),
            logger
        )

        # 2. Enhanced Age Distribution Analysis
        age_statistics = create_enhanced_age_distribution(df, step3_plots_folder, logger)
        statistics["age"] = age_statistics

        # 3. Enhanced Visit Pattern Analysis
        visit_statistics = analyze_visit_patterns(df, step3_plots_folder, logger)
        statistics["visits"] = visit_statistics

        # 4. Gender Distribution Analysis
        gender_statistics = analyze_gender_distribution(df, step3_plots_folder, logger)
        statistics["gender"] = gender_statistics

        # Export all statistics to Excel
        export_path = export_statistics_to_excel(statistics, step3_output_folder, logger)

        # Generate a summary report in Markdown
        summary_path = os.path.join(step3_output_folder, "general_analysis_summary.md")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Erweiterte Allgemeine Analyse Zusammenfassung\n\n")
            f.write(f"**Datum der Analyse:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Übersicht der Daten\n")
            f.write(f"- **Gesamtzahl der Datensätze:** {statistics['total_records']}\n")
            f.write(f"- **Eindeutige Fälle:** {statistics['distinct_cases']}\n")
            f.write(f"- **Anzahl der Spalten:** {statistics['columns_count']}\n\n")

            if age_statistics:
                f.write("## Altersanalyse\n")
                f.write(f"- **Durchschnittsalter:** {age_statistics['mean_age']:.1f} Jahre\n")
                f.write(f"- **Medianalter:** {age_statistics['median_age']:.1f} Jahre\n")
                f.write(f"- **Altersbereich:** {age_statistics['min_age']} bis {age_statistics['max_age']} Jahre\n")
                f.write(f"- **Standardabweichung:** {age_statistics['std_dev_age']:.1f} Jahre\n")
                f.write(f"- **Schiefe:** {age_statistics['skewness']:.2f} (")

                if age_statistics['skewness'] > 0.5:
                    f.write("rechtsschiefe Verteilung - mehr jüngere Patienten)\n")
                elif age_statistics['skewness'] < -0.5:
                    f.write("linksschiefe Verteilung - mehr ältere Patienten)\n")
                else:
                    f.write("annähernd symmetrische Verteilung)\n")

                f.write(f"- **Verteilung:** ")
                if age_statistics['normality_p_value'] > 0.05:
                    f.write("Normalverteilt (p > 0.05)\n\n")
                else:
                    f.write(f"Nicht normalverteilt (p = {age_statistics['normality_p_value']:.4f})\n\n")

            if visit_statistics:
                f.write("## Analyse der Besuchsmuster\n")
                vs = visit_statistics['visits_statistics']
                f.write(
                    f"- **Besuche pro Patient:** {vs['min_visits']} bis {vs['max_visits']} (Durchschnitt: {vs['avg_visits']:.1f})\n")

                if 'timing_statistics' in visit_statistics:
                    ts = visit_statistics['timing_statistics']
                    f.write(
                        f"- **Zeitpunkt des ersten Besuchs:** {ts['first_visit_min']} bis {ts['first_visit_max']} Tage nach Unfall (Mittelwert: {ts['first_visit_mean']:.1f})\n")
                    f.write(
                        f"- **Zeitpunkt des letzten Besuchs:** {ts['last_visit_min']} bis {ts['last_visit_max']} Tage nach Unfall (Mittelwert: {ts['last_visit_mean']:.1f})\n")
                    f.write(
                        f"- **Heilungsdauer:** {ts['healing_duration_min']} bis {ts['healing_duration_max']} Tage (Mittelwert: {ts['healing_duration_mean']:.1f})\n")
                    f.write(f"- **Korrelation zwischen erstem und letztem Besuch:** {ts['correlation_first_last']:.2f}")

                    if ts['correlation_p_value'] < 0.05:
                        f.write(" (statistisch signifikant)\n\n")
                    else:
                        f.write(f" (nicht statistisch signifikant, p = {ts['correlation_p_value']:.4f})\n\n")

            if gender_statistics:
                f.write("## Geschlechterverteilung\n")
                for gender, count in gender_statistics['gender_counts'].items():
                    percentage = gender_statistics['gender_percentages'][gender]
                    f.write(f"- **{gender}:** {count} Patienten ({percentage:.1f}%)\n")

            f.write("\n## Erzeugte Visualisierungen\n")
            for root, _, files in os.walk(step3_plots_folder):
                for file in files:
                    if file.endswith('.png'):
                        rel_path = os.path.join(os.path.relpath(root, os.path.dirname(step3_output_folder)), file)
                        f.write(f"- [{file}]({rel_path})\n")

            f.write("\n## Zusätzliche Ressourcen\n")
            f.write(f"- [Vollständige Statistik in Excel](general_analysis_statistics.xlsx)\n")
            f.write("\n*Dieser Bericht wurde automatisch im Rahmen der erweiterten Analyse erstellt.*")

        logger.info(f"Summary report saved to: {summary_path}")
        logger.info("Enhanced general analysis completed successfully.")

    except Exception as e:
        logger.error(f"Error in general analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    general_analysis()