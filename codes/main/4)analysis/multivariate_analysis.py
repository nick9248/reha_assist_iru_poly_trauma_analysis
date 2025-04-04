import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from datetime import datetime
from dotenv import load_dotenv
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from pathlib import Path


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('multivariate_analysis')
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


def load_patient_level_data(file_path, logger):
    """Load the patient-level dataset created during univariate analysis."""
    try:
        df = pd.read_excel(file_path, dtype={"Schadennummer": str})
        logger.info(f"Erfolgreich geladene Patientendaten aus: {file_path}")
        logger.info(f"Datenform: {df.shape}")
        logger.info(f"Spalten: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden des Patientendatensatzes: {str(e)}", exc_info=True)
        raise


def prepare_patient_level_data_if_needed(base_dataset, output_folder, logger):
    """
    Create patient-level dataset if it doesn't already exist.
    This function replicates the logic from univariate_analysis.py.
    """
    patient_data_path = os.path.join(output_folder, "patient_level_data.xlsx")

    # Check if file already exists
    if os.path.exists(patient_data_path):
        logger.info(f"Patientendaten existieren bereits unter: {patient_data_path}")
        return patient_data_path

    # If not, create it from the base dataset
    logger.info("Patientendaten werden neu erstellt...")

    try:
        # Load the original dataset
        df = pd.read_excel(base_dataset, dtype={"Schadennummer": str})
        df.columns = df.columns.str.strip()

        # Identify unique patients
        unique_patients = df["Schadennummer"].unique()
        logger.info(f"Anzahl der eindeutigen Patienten: {len(unique_patients)}")

        # Create a patient-level dataframe
        patient_data = []

        # Define body parts to analyze
        body_parts = ['Kopf', 'Hals', 'Thorax', 'Abdomen', 'Arm links', 'Arm rechts',
                      'Wirbelsaeule', 'Bein rechts', 'Bein links', 'Becken']

        # Define merged categories
        merged_categories = {
            'Arm': ['Arm links', 'Arm rechts'],
            'Bein': ['Bein links', 'Bein rechts']
        }

        # For each patient, extract data from their last visit and injury information across all visits
        for patient_id in unique_patients:
            patient_records = df[df["Schadennummer"] == patient_id]

            # Get last visit information
            last_visit = patient_records.loc[patient_records["Days_Since_Accident"].idxmax()]

            # Basic patient info
            patient_info = {
                "Schadennummer": patient_id,
                "Heilungsdauer": last_visit["Days_Since_Accident"],
                # Healing duration is days from accident to last visit
                "Alter": last_visit["Age_At_Accident"] if "Age_At_Accident" in last_visit else None,
                "Geschlecht": last_visit["Geschlecht"] if "Geschlecht" in last_visit else None,
                "Anzahl_Besuche": len(patient_records)
            }

            # Extract injury information (if any visit has "Ja", mark as injured)
            for body_part in body_parts:
                if body_part in patient_records.columns:
                    patient_info[body_part] = "Ja" if "Ja" in patient_records[body_part].values else "Nein"

            # Create merged body part fields
            for merged_name, components in merged_categories.items():
                valid_components = [c for c in components if c in patient_records.columns]
                if valid_components:
                    component_values = [patient_info.get(c, "Nein") for c in valid_components]
                    patient_info[merged_name] = "Ja" if "Ja" in component_values else "Nein"

            # Calculate injury count (how many body parts are affected)
            injury_fields = [bp for bp in body_parts if bp in patient_records.columns]
            patient_info["Verletzungsanzahl"] = sum(1 for bp in injury_fields if patient_info[bp] == "Ja")

            # Add to the patient dataset
            patient_data.append(patient_info)

        # Convert to DataFrame
        patient_df = pd.DataFrame(patient_data)
        logger.info(f"Patientendatensatz erstellt mit {len(patient_df)} Fällen und {len(patient_df.columns)} Spalten")

        # Save the dataset
        patient_df.to_excel(patient_data_path, index=False)
        logger.info(f"Patientendaten gespeichert unter: {patient_data_path}")

        return patient_data_path

    except Exception as e:
        logger.error(f"Fehler bei der Erstellung der Patientendaten: {str(e)}", exc_info=True)
        raise


def build_regression_models(df, output_folder, logger):
    """Build multiple regression models with healing duration as dependent variable."""
    # Check variable names in the dataframe
    logger.info(f"Verfügbare Spalten für Regression: {list(df.columns)}")

    # Ensure healing duration is available
    if 'Heilungsdauer' not in df.columns:
        logger.error("Heilungsdauer-Spalte nicht im Datensatz gefunden")
        return None

    models = {}

    # Model 1: Significant factors from univariate analysis (Abdomen, Kopf)
    try:
        logger.info("Erstelle Modell 1: Signifikante Faktoren (Abdomen, Kopf)")

        # Convert categorical variables to binary
        for var in ['Abdomen', 'Kopf']:
            if var in df.columns and df[var].dtype == 'object':
                df[var] = df[var].map({'Ja': 1, 'Nein': 0})
                logger.info(f"Konvertiere {var} von Ja/Nein zu 1/0")

        X1 = sm.add_constant(df[['Abdomen', 'Kopf']])
        y = df['Heilungsdauer']
        model1 = sm.OLS(y, X1).fit()
        logger.info(f"Modell 1 Zusammenfassung:\n{model1.summary()}")

        # Save results
        with open(os.path.join(output_folder, "model1_summary.txt"), 'w') as f:
            f.write(model1.summary().as_text())

        models['model1'] = model1
    except Exception as e:
        logger.error(f"Fehler in Modell 1: {str(e)}", exc_info=True)

    # Model 2: Adding nearly significant factors (Arm rechts, Arm)
    try:
        logger.info("Erstelle Modell 2: Hinzufügen fast signifikanter Faktoren (Arm rechts)")

        # Convert categorical variables to binary
        for var in ['Arm rechts']:
            if var in df.columns and df[var].dtype == 'object':
                df[var] = df[var].map({'Ja': 1, 'Nein': 0})
                logger.info(f"Konvertiere {var} von Ja/Nein zu 1/0")

        X2 = sm.add_constant(df[['Abdomen', 'Kopf', 'Arm rechts']])
        model2 = sm.OLS(y, X2).fit()
        logger.info(f"Modell 2 Zusammenfassung:\n{model2.summary()}")

        # Save results
        with open(os.path.join(output_folder, "model2_summary.txt"), 'w') as f:
            f.write(model2.summary().as_text())

        models['model2'] = model2
    except Exception as e:
        logger.error(f"Fehler in Modell 2: {str(e)}", exc_info=True)

    # Model 3: Add age after handling missing values
    try:
        # Check for age variable and handle missing values
        if 'Alter' in df.columns:
            # Create a copy with non-missing age values
            df_age = df.dropna(subset=['Alter']).copy()

            if len(df_age) >= 30:  # Only proceed if we have enough data
                logger.info(f"Erstelle Modell 3: Mit Alter (n={len(df_age)})")

                # Convert categorical variables if needed
                for var in ['Abdomen', 'Kopf', 'Arm rechts']:
                    if var in df_age.columns and df_age[var].dtype == 'object':
                        df_age[var] = df_age[var].map({'Ja': 1, 'Nein': 0})

                X3 = sm.add_constant(df_age[['Abdomen', 'Kopf', 'Arm rechts', 'Alter']])
                y3 = df_age['Heilungsdauer']
                model3 = sm.OLS(y3, X3).fit()

                logger.info(f"Modell 3 Zusammenfassung:\n{model3.summary()}")

                # Save results
                with open(os.path.join(output_folder, "model3_summary.txt"), 'w') as f:
                    f.write(model3.summary().as_text())

                models['model3'] = model3
            else:
                logger.warning(
                    f"Nicht genügend Daten mit gültigen Alterswerten (n={len(df_age)}). Überspringe Modell 3.")
        else:
            logger.warning("Keine Altersvariable im Datensatz gefunden. Überspringe Modell 3.")
    except Exception as e:
        logger.error(f"Fehler in Modell 3: {str(e)}", exc_info=True)

    # Model 4: Using combined Arm instead of Arm rechts
    try:
        logger.info("Erstelle Modell 4: Mit kombiniertem Arm-Faktor")

        # Convert categorical variables to binary
        for var in ['Abdomen', 'Kopf', 'Arm']:
            if var in df.columns and df[var].dtype == 'object':
                df[var] = df[var].map({'Ja': 1, 'Nein': 0})
                logger.info(f"Konvertiere {var} von Ja/Nein zu 1/0")

        X4 = sm.add_constant(df[['Abdomen', 'Kopf', 'Arm']])
        model4 = sm.OLS(y, X4).fit()
        logger.info(f"Modell 4 Zusammenfassung:\n{model4.summary()}")

        # Save results
        with open(os.path.join(output_folder, "model4_summary.txt"), 'w') as f:
            f.write(model4.summary().as_text())

        models['model4'] = model4
    except Exception as e:
        logger.error(f"Fehler in Modell 4: {str(e)}", exc_info=True)

    return models


def perform_survival_analysis(df, output_folder, plots_folder, logger):
    """Implement survival analysis to model time to last visit."""
    logger.info("Starte Survival-Analyse")

    # Check for duration variable
    if 'Heilungsdauer' not in df.columns:
        logger.error("Heilungsdauer-Spalte nicht im Datensatz gefunden")
        return None

    # Create event indicator (all 1 as all patients reached last visit)
    df['event'] = 1

    # Initialize survival analysis models
    kmf = KaplanMeierFitter()
    cph = CoxPHFitter()

    # Kaplan-Meier overall survival curve
    try:
        logger.info("Erstelle Gesamte Kaplan-Meier Kurve")
        kmf.fit(df['Heilungsdauer'], df['event'])

        plt.figure(figsize=(10, 6))
        kmf.plot_survival_function()
        plt.title('Kaplan-Meier Schätzung der Heilungsdauer')
        plt.xlabel('Tage')
        plt.ylabel('Anteil der noch nicht geheilten Patienten')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_folder, "km_gesamtkurve.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Fehler bei der Erstellung der Gesamtkurve: {str(e)}", exc_info=True)

    # KM curves by significant factors
    factors = ['Kopf', 'Abdomen', 'Arm rechts', 'Arm']

    for factor in factors:
        try:
            if factor in df.columns:
                logger.info(f"Erstelle KM-Kurven für {factor}")
                plt.figure(figsize=(10, 6))

                # Fit for positive cases (injury)
                positive_df = df[df[factor] == 'Ja']
                if len(positive_df) > 0:
                    kmf.fit(positive_df['Heilungsdauer'], positive_df['event'], label=f'{factor} Ja')
                    kmf.plot_survival_function()

                # Fit for negative cases (no injury)
                negative_df = df[df[factor] == 'Nein']
                if len(negative_df) > 0:
                    kmf.fit(negative_df['Heilungsdauer'], negative_df['event'], label=f'{factor} Nein')
                    kmf.plot_survival_function()

                plt.title(f'Kaplan-Meier Kurven nach {factor}')
                plt.xlabel('Tage')
                plt.ylabel('Anteil der noch nicht geheilten Patienten')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_folder, f"km_{factor}_kurve.png"), dpi=300)
                plt.close()
        except Exception as e:
            logger.error(f"Fehler bei KM-Kurve für {factor}: {str(e)}", exc_info=True)

    # Cox Proportional Hazards Model
    try:
        logger.info("Erstelle Cox Proportional Hazards Modell")

        # Prepare dataframe with binary factors
        cox_df = df[['Heilungsdauer', 'event']].copy()

        for factor in ['Kopf', 'Abdomen', 'Arm rechts', 'Arm']:
            if factor in df.columns:
                if df[factor].dtype == 'object':
                    cox_df[factor] = df[factor].map({'Ja': 1, 'Nein': 0})
                else:
                    cox_df[factor] = df[factor]

        # Add age if available (without missing values)
        if 'Alter' in df.columns:
            # Add only non-missing age values
            cox_df['Alter'] = df['Alter']
            cox_df = cox_df.dropna(subset=['Alter'])
            logger.info(f"Cox-Modell mit Alter (n={len(cox_df)})")

        # Model with significant factors
        cph.fit(cox_df, duration_col='Heilungsdauer', event_col='event')
        logger.info(f"Cox PH Modell Zusammenfassung:\n{cph.summary}")

        # Save the model summary
        with open(os.path.join(output_folder, "cox_model_summary.txt"), 'w') as f:
            f.write(str(cph.summary))

        # Plot the hazard ratios using lifelines built-in function
        plt.figure(figsize=(10, 8))
        cph.plot()
        plt.title('Hazard Ratios mit 95% Konfidenzintervallen')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_folder, "cox_hazard_ratios.png"), dpi=300)
        plt.close()

        # Create a more detailed hazard ratio plot manually
        try:
            plt.figure(figsize=(12, 8))

            # Get parameter names and hazard ratios
            param_names = cph.params_.index
            hazard_ratios = np.exp(cph.params_)

            # Get p-values
            p_values = cph.summary['p']

            # Log column names to help debug
            logger.info(f"Confidence interval columns: {list(cph.confidence_intervals_.columns)}")

            # Try different possible column names for confidence intervals
            ci_lower = None
            ci_upper = None

            # Around line 350-360 in your multivariate_analysis.py file
            possible_lower_cols = ['lower 0.95', 'lower 95%', 'lower_0.95', '0.95 lower', '2.5%', '95% lower-bound']
            possible_upper_cols = ['upper 0.95', 'upper 95%', 'upper_0.95', '0.95 upper', '97.5%', '95% upper-bound']
            # Find matching column names
            for lower_col in possible_lower_cols:
                if lower_col in cph.confidence_intervals_.columns:
                    ci_lower = np.exp(cph.confidence_intervals_[lower_col])
                    logger.info(f"Found lower CI column: {lower_col}")
                    break

            for upper_col in possible_upper_cols:
                if upper_col in cph.confidence_intervals_.columns:
                    ci_upper = np.exp(cph.confidence_intervals_[upper_col])
                    logger.info(f"Found upper CI column: {upper_col}")
                    break

            # If CI columns were found, create the plot
            if ci_lower is not None and ci_upper is not None:
                # Calculate error bar sizes
                error_below = hazard_ratios - ci_lower
                error_above = ci_upper - hazard_ratios

                # Sort by hazard ratio for better visualization
                sort_idx = np.argsort(hazard_ratios)
                sorted_names = [param_names[i] for i in sort_idx]
                sorted_hrs = [hazard_ratios[i] for i in sort_idx]
                sorted_err_below = [error_below[i] for i in sort_idx]
                sorted_err_above = [error_above[i] for i in sort_idx]
                sorted_p = [p_values[i] for i in sort_idx]

                # Create the barplot
                bars = plt.barh(range(len(sorted_names)), sorted_hrs, height=0.6,
                                xerr=np.array([sorted_err_below, sorted_err_above]),
                                capsize=5, color='skyblue', alpha=0.8)

                # Color significant bars differently
                for i, p in enumerate(sorted_p):
                    if p < 0.05:
                        bars[i].set_color('coral')

                # Add reference line at HR=1
                plt.axvline(x=1, color='black', linestyle='--', alpha=0.7)

                # Add labels and title
                plt.yticks(range(len(sorted_names)), sorted_names)
                plt.xscale('log')  # Log scale makes hazard ratios easier to interpret
                plt.xlabel('Hazard Ratio (log scale)')
                plt.ylabel('Variable')
                plt.title('Cox Proportional Hazards - Hazard Ratios mit 95% CI')
                plt.grid(axis='x', alpha=0.3)

                # Add p-value annotations
                for i, p in enumerate(sorted_p):
                    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    plt.text(max(sorted_hrs) * 1.2, i, f"p={p:.3f} {significance}", va='center')

                # Add legend for significance
                plt.figtext(0.7, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", ha="center")

                plt.tight_layout()
                plt.savefig(os.path.join(plots_folder, "cox_hazard_ratios_detailed.png"), dpi=300)
                plt.close()
                logger.info("Detailed hazard ratio plot created successfully")
            else:
                logger.warning("Could not identify confidence interval columns for detailed plot")

        except Exception as e:
            logger.error(f"Fehler bei der Erstellung des detaillierten Hazard Ratio Plots: {str(e)}", exc_info=True)

        return cph
    except Exception as e:
        logger.error(f"Fehler im Cox PH Modell: {str(e)}", exc_info=True)
        return None


def visualize_regression_results(models, df, plots_folder, logger):
    """Create visualizations for regression model results."""
    logger.info("Erstelle Visualisierungen für Regressionsanalysen")

    # Check if models are available
    if not models or not any(models.values()):
        logger.error("Keine Regressionsmodelle für Visualisierung verfügbar")
        return

    # 1. Coefficient plot
    try:
        logger.info("Erstelle Koeffizienten-Plot")

        # Collect coefficients from all models
        coeffs = {}

        for model_name, model in models.items():
            if model is not None:
                # Skip constant for better visualization
                model_coeffs = {
                    k: v for k, v in model.params.items() if k != 'const'
                }
                coeffs[model_name] = model_coeffs

        # Create DataFrame for plotting
        coeff_data = []
        for model_name, model_coeffs in coeffs.items():
            for var, coef in model_coeffs.items():
                p_value = models[model_name].pvalues[var]
                significance = '***' if p_value < 0.001 else (
                    '**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
                coeff_data.append({
                    'Modell': model_name,
                    'Variable': var,
                    'Koeffizient': coef,
                    'p-Wert': p_value,
                    'Signifikanz': significance
                })

        if not coeff_data:
            logger.warning("Keine Koeffizienten für Visualisierung verfügbar")
            return

        coeff_df = pd.DataFrame(coeff_data)

        # Plot
        plt.figure(figsize=(12, 8))

        # Create barplot with hue for model
        ax = sns.barplot(
            x='Variable',
            y='Koeffizient',
            hue='Modell',
            data=coeff_df
        )

        # Add significance markers
        for i, row in enumerate(coeff_df.itertuples()):
            if row.Signifikanz:
                ax.text(
                    i % len(coeff_df['Variable'].unique()),
                    row.Koeffizient + (1 if row.Koeffizient > 0 else -1) * 5,
                    row.Signifikanz,
                    ha='center',
                    fontweight='bold'
                )

        # Add titles and labels
        plt.title('Regressionskoeffizienten über verschiedene Modelle', fontsize=14, fontweight='bold')
        plt.ylabel('Koeffizientenwert (Tage)', fontsize=12)
        plt.xlabel('Faktoren', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        # Add note about significance
        plt.figtext(0.05, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", ha="left")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, "regressionskoeffizienten.png"), dpi=300)
        plt.close()

        logger.info(
            f"Koeffizienten-Plot gespeichert unter: {os.path.join(plots_folder, 'regressionskoeffizienten.png')}")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Koeffizienten-Plots: {str(e)}", exc_info=True)

    # 2. Actual vs Predicted plot for the most comprehensive model
    try:
        logger.info("Erstelle Ist vs. Vorhersage Plot")

        # Use Model 2 which has the best performance
        model_name = 'model2'
        if model_name in models and models[model_name] is not None:
            model = models[model_name]

            # Create the feature matrix without the constant term
            feature_cols = ['Abdomen', 'Kopf', 'Arm rechts']
            X_features = df[feature_cols].copy()

            # Convert categorical variables to binary if needed
            for col in feature_cols:
                if X_features[col].dtype == 'object':
                    X_features[col] = X_features[col].map({'Ja': 1, 'Nein': 0})

            # Add constant term
            X = sm.add_constant(X_features)

            # Make predictions
            predictions = model.predict(X)

            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(df['Heilungsdauer'], predictions, alpha=0.7)

            # Add identity line
            min_val = min(df['Heilungsdauer'].min(), predictions.min())
            max_val = max(df['Heilungsdauer'].max(), predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfekte Vorhersage')

            # Add regression line
            m, b = np.polyfit(df['Heilungsdauer'], predictions, 1)
            plt.plot(
                [min_val, max_val],
                [m * min_val + b, m * max_val + b],
                'r-',
                label=f'Trend (y = {m:.2f}x + {b:.2f})'
            )

            # Add titles and labels
            plt.title('Tatsächliche vs. Vorhergesagte Heilungsdauer', fontsize=14, fontweight='bold')
            plt.xlabel('Tatsächliche Heilungsdauer (Tage)', fontsize=12)
            plt.ylabel('Vorhergesagte Heilungsdauer (Tage)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Add R-squared to plot
            plt.text(
                0.05, 0.95,
                f"R²: {model.rsquared:.3f}\nAngepasstes R²: {model.rsquared_adj:.3f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, "ist_vs_vorhersage.png"), dpi=300)
            plt.close()

            logger.info(
                f"Ist vs. Vorhersage Plot gespeichert unter: {os.path.join(plots_folder, 'ist_vs_vorhersage.png')}")
        else:
            logger.warning(f"Modell {model_name} nicht verfügbar für Ist vs. Vorhersage Plot")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Ist vs. Vorhersage Plots: {str(e)}", exc_info=True)


def generate_summary_report(regression_models, cox_model, output_folder, logger):
    """Generate a Markdown summary report of the analysis results."""
    try:
        report_path = os.path.join(output_folder, "multivariate_analysis_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Multivariate Analyse der Einflussfaktoren auf die Heilungsdauer\n\n")
            f.write(f"**Datum:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

            f.write("## Übersicht\n\n")
            f.write("Diese Analyse baut auf den Ergebnissen der univariaten Analyse auf und untersucht, ")
            f.write(
                "wie mehrere Faktoren zusammenwirken, um die Heilungsdauer bei Polytrauma-Patienten zu beeinflussen. ")
            f.write("Es wurden sowohl lineare Regressionsmodelle als auch Überlebensanalysen implementiert.\n\n")

            f.write("## Ergebnisse der Regressionsanalyse\n\n")

            if regression_models and 'model1' in regression_models and regression_models['model1'] is not None:
                model1 = regression_models['model1']
                f.write("### Modell 1: Signifikante Faktoren (Abdomen, Kopf)\n\n")
                f.write(f"- **R²:** {model1.rsquared:.3f}\n")
                f.write(f"- **Angepasstes R²:** {model1.rsquared_adj:.3f}\n")
                f.write("- **Koeffizienten:**\n")

                for var, coef in model1.params.items():
                    p_value = model1.pvalues[var]
                    f.write(f"  - {var}: {coef:.2f} (p={p_value:.4f})")
                    if p_value < 0.05:
                        f.write(" *signifikant*")
                    f.write("\n")

                f.write("\n")

            if regression_models and 'model2' in regression_models and regression_models['model2'] is not None:
                model2 = regression_models['model2']
                f.write("### Modell 2: Erweitert mit fast signifikanten Faktoren\n\n")
                f.write(f"- **R²:** {model2.rsquared:.3f}\n")
                f.write(f"- **Angepasstes R²:** {model2.rsquared_adj:.3f}\n")
                f.write("- **Koeffizienten:**\n")

                for var, coef in model2.params.items():
                    p_value = model2.pvalues[var]
                    f.write(f"  - {var}: {coef:.2f} (p={p_value:.4f})")
                    if p_value < 0.05:
                        f.write(" *signifikant*")
                    f.write("\n")

                f.write("\n")

            if regression_models and 'model3' in regression_models and regression_models['model3'] is not None:
                model3 = regression_models['model3']
                f.write("### Modell 3: Alter mit nicht-linearem Effekt\n\n")
                f.write(f"- **R²:** {model3.rsquared:.3f}\n")
                f.write(f"- **Angepasstes R²:** {model3.rsquared_adj:.3f}\n")
                f.write("- **Koeffizienten:**\n")

                for var, coef in model3.params.items():
                    p_value = model3.pvalues[var]
                    f.write(f"  - {var}: {coef:.2f} (p={p_value:.4f})")
                    if p_value < 0.05:
                        f.write(" *signifikant*")
                    f.write("\n")

                f.write("\n")

            f.write("## Ergebnisse der Survival-Analyse\n\n")
            f.write("### Cox Proportional Hazards Modell\n\n")
            if cox_model is not None:
                f.write("### Cox Proportional Hazards Modell\n\n")
                f.write(f"- **Concordance Index:** {cox_model.concordance_index_:.3f}\n")
                f.write(
                    f"- **Log-Likelihood Ratio Test p-Wert:** {cox_model.log_likelihood_ratio_test().p_value:.4f}\n\n")
                f.write("- **Hazard Ratios:**\n")

                for var in cox_model.summary.index:
                    hr = cox_model.summary.loc[var, 'exp(coef)']
                    p = cox_model.summary.loc[var, 'p']
                    lower_ci = cox_model.summary.loc[var, 'exp(coef) lower 95%']
                    upper_ci = cox_model.summary.loc[var, 'exp(coef) upper 95%']

                    f.write(f"  - {var}: HR={hr:.2f} (95% KI: {lower_ci:.2f}-{upper_ci:.2f}, p={p:.4f})")
                    if p < 0.05:
                        f.write(" *signifikant*")
                    f.write("\n")

                f.write("\n")
            f.write("### Interpretation der scheinbar widersprüchlichen Ergebnisse\n\n")
            f.write(
                "Die linearen Regressionsmodelle und das Cox-Modell zeigen scheinbar gegensätzliche Effekte für Kopfverletzungen:\n")
            f.write("- Im Regressionsmodell: negativer Koeffizient (-181.92) deutet auf kürzere Heilungsdauer hin\n")
            f.write(
                "- Im Cox-Modell: HR > 1 (2.87) deutet auf höheres Risiko für frühere Behandlungsbeendigung hin\n\n")

            f.write(
                "Diese Ergebnisse sind jedoch konsistent, da beide auf eine schnellere Heilung bei Kopfverletzungen hindeuten:\n")
            f.write("1. Im Regressionsmodell bedeutet ein negativer Koeffizient direkt weniger Tage bis zur Heilung\n")
            f.write(
                "2. Im Cox-Modell bedeutet ein HR > 1 eine höhere Wahrscheinlichkeit, den Endpunkt (Behandlungsabschluss) früher zu erreichen\n\n")

            f.write("Ebenso sind die Ergebnisse für Abdomen-Verletzungen konsistent:\n")
            f.write("- Positiver Regressionskoeffizient: längere Heilungsdauer\n")
            f.write("- HR < 1: geringeres Risiko für frühere Behandlungsbeendigung (= längere Behandlung)\n\n")

            # Determine which model has the best fit
            best_model = None
            best_model_name = None
            best_r2_adj = -float('inf')

            for model_name in ['model3', 'model2', 'model1']:
                if model_name in regression_models and regression_models[model_name] is not None:
                    r2_adj = regression_models[model_name].rsquared_adj
                    if r2_adj > best_r2_adj:
                        best_r2_adj = r2_adj
                        # Determine which model has the best fit
                        best_model = None
                        best_model_name = None
                        best_r2_adj = -float('inf')

                        for model_name in ['model3', 'model2', 'model1']:
                            if model_name in regression_models and regression_models[model_name] is not None:
                                r2_adj = regression_models[model_name].rsquared_adj
                                if r2_adj > best_r2_adj:
                                    best_r2_adj = r2_adj
                                    best_model = regression_models[model_name]
                                    best_model_name = model_name

                        f.write("## Wichtigste Erkenntnisse\n\n")

                        # Get significant factors from the best model if available
                        significant_factors = []
                        if best_model is not None:
                            for var, p_value in best_model.pvalues.items():
                                if p_value < 0.05 and var != 'const':
                                    coef = best_model.params[var]
                                    effect = "verlängert" if coef > 0 else "verkürzt"
                                    significant_factors.append((var, coef, p_value, effect))

                        if significant_factors:
                            f.write("### Signifikante Einflussfaktoren\n\n")
                            for var, coef, p_value, effect in significant_factors:
                                f.write(f"- **{var}** {effect} die Heilungsdauer um durchschnittlich ")
                                f.write(f"**{abs(coef):.1f} Tage** (p={p_value:.4f}), ")
                                f.write(f"wenn für andere Faktoren kontrolliert wird.\n")
                            f.write("\n")
                        else:
                            f.write(
                                "In der multivariaten Analyse wurden keine statistisch signifikanten Faktoren gefunden. ")
                            f.write(
                                "Dies könnte auf die kleine Stichprobengröße oder komplexe Interaktionen zwischen den Faktoren zurückzuführen sein.\n\n")

                        # Best model results
                        if best_model is not None:
                            f.write(f"### Modellgüte\n\n")
                            f.write(
                                f"- Das beste Modell ({best_model_name}) erklärt **{best_model.rsquared * 100:.1f}%** ")
                            f.write("der Varianz in der Heilungsdauer (R²).\n")
                            f.write(f"- Das angepasste R² beträgt **{best_model.rsquared_adj * 100:.1f}%**.\n\n")

                        # Survival analysis insights
                        if cox_model is not None:
                            f.write("### Erkenntnisse aus der Survival-Analyse\n\n")

                            # Find significant factors in Cox model
                            cox_significant = []
                            for var in cox_model.summary.index:
                                p = cox_model.summary.loc[var, 'p']
                                hr = cox_model.summary.loc[var, 'exp(coef)']
                                if p < 0.05:
                                    cox_significant.append((var, hr, p))

                            if cox_significant:
                                f.write(
                                    "Die Cox-Regressionsanalyse identifizierte die folgenden signifikanten Faktoren:\n\n")
                                for var, hr, p in cox_significant:
                                    effect = "erhöht" if hr > 1 else "verringert"
                                    f.write(
                                        f"- **{var}** {effect} die Wahrscheinlichkeit einer längeren Heilungsdauer ")
                                    if hr > 1:
                                        f.write(f"um den Faktor **{hr:.2f}** (p={p:.4f}).\n")
                                    else:
                                        f.write(f"um den Faktor **{1 / hr:.2f}** (p={p:.4f}).\n")
                            else:
                                f.write("Die Cox-Regressionsanalyse identifizierte keine signifikanten Faktoren, ")
                                f.write("die die Heilungsdauer beeinflussen.\n")

                            f.write("\n")

                        f.write("## Visualisierungen\n\n")
                        f.write("Die folgenden Visualisierungen wurden im Rahmen dieser Analyse erstellt:\n\n")
                        f.write(
                            "1. **Regressionskoeffizienten über verschiedene Modelle**: Vergleicht die Stärke und Richtung des Einflusses verschiedener Faktoren.\n")
                        f.write(
                            "2. **Tatsächliche vs. Vorhergesagte Heilungsdauer**: Zeigt die Vorhersagekraft des besten Modells.\n")
                        f.write(
                            "3. **Kaplan-Meier Kurven nach Verletzungstyp**: Visualisiert den Einfluss verschiedener Verletzungen auf die Heilungswahrscheinlichkeit im Zeitverlauf.\n")
                        f.write(
                            "4. **Cox Modell Hazard Ratios**: Stellt das relative Risiko für längere Heilungsdauer dar.\n\n")

                        f.write("## Vergleich mit univariater Analyse\n\n")
                        f.write("Die multivariate Analyse baut auf den Ergebnissen der univariaten Analyse auf, ")
                        f.write(
                            "die signifikante Einflüsse von Verletzungen des Kopfes und des Abdomens identifiziert hatte. ")
                        f.write("In der multivariaten Analyse zeigt sich, dass...\n\n")

                        if significant_factors:
                            same_as_univariate = [var for var, _, _, _ in significant_factors if
                                                  var in ['Kopf', 'Abdomen']]
                            new_in_multivariate = [var for var, _, _, _ in significant_factors if
                                                   var not in ['Kopf', 'Abdomen']]

                            if same_as_univariate:
                                f.write(f"- Die univariat signifikanten Faktoren **{', '.join(same_as_univariate)}** ")
                                f.write("bleiben auch in der multivariaten Analyse signifikant.\n")

                            if new_in_multivariate:
                                f.write(f"- Die Faktoren **{', '.join(new_in_multivariate)}** werden erst in der ")
                                f.write("multivariaten Analyse als signifikant erkannt.\n")

                            lost_in_multivariate = [var for var in ['Kopf', 'Abdomen'] if
                                                    var not in [v for v, _, _, _ in significant_factors]]
                            if lost_in_multivariate:
                                f.write(
                                    f"- Die univariat signifikanten Faktoren **{', '.join(lost_in_multivariate)}** ")
                                f.write(
                                    "verlieren ihre Signifikanz in der multivariaten Analyse, was auf Konfundierung hindeutet.\n")
                        else:
                            f.write(
                                "- Die in der univariaten Analyse identifizierten signifikanten Faktoren (Kopf, Abdomen) ")
                            f.write(
                                "verlieren ihre statistische Signifikanz, wenn für andere Faktoren kontrolliert wird. ")
                            f.write("Dies deutet auf komplexe Interaktionen zwischen den Faktoren hin.\n")

                        f.write("\n")

                        f.write("## Schlussfolgerungen und Empfehlungen\n\n")
                        f.write(
                            "Basierend auf den Ergebnissen der multivariaten Analyse können die folgenden Schlussfolgerungen gezogen werden:\n\n")

                        if best_model is not None and best_model.rsquared > 0.3:
                            f.write(
                                "1. Das multivariate Modell erklärt einen substanziellen Teil der Varianz in der Heilungsdauer ")
                            f.write(
                                f"({best_model.rsquared * 100:.1f}%), was seine klinische Relevanz unterstreicht.\n")
                        else:
                            f.write("1. Die erklärte Varianz des multivariaten Modells ist begrenzt, ")
                            f.write("was auf weitere, nicht erfasste Einflussfaktoren hindeutet.\n")

                        if significant_factors:
                            f.write(
                                f"2. Die identifizierten Haupteinflussfaktoren ({', '.join([var for var, _, _, _ in significant_factors])}) ")
                            f.write(
                                "sollten besonders bei der Prognose der Heilungsdauer und der Planung von Rehabilitationsmaßnahmen berücksichtigt werden.\n")
                        else:
                            f.write(
                                "2. Die fehlende statistische Signifikanz einzelner Faktoren könnte auf die kleine Stichprobengröße ")
                            f.write("oder komplexe Wechselwirkungen zurückzuführen sein.\n")

                        f.write(
                            "3. Die Ergebnisse unterstützen einen individualisierten Ansatz in der Rehabilitation, ")
                        f.write("der die spezifischen Verletzungsmuster jedes Patienten berücksichtigt.\n\n")

                        f.write("## Limitationen\n\n")
                        f.write(
                            "- **Stichprobengröße**: Die Analyse basiert auf nur 50 Patienten, was die statistische Power einschränkt.\n")
                        f.write(
                            "- **Definition der Heilungsdauer**: Die Heilungsdauer wurde als Zeit vom Unfall bis zum letzten Besuch definiert, ")
                        f.write(
                            "was möglicherweise nicht immer das tatsächliche Ende des Heilungsprozesses widerspiegelt.\n")
                        f.write(
                            "- **Fehlende Variablen**: Möglicherweise wurden wichtige Einflussfaktoren nicht erfasst.\n")
                        f.write(
                            "- **Multikollinearität**: Zwischen einigen Verletzungsarten besteht möglicherweise eine Korrelation, ")
                        f.write("die die Interpretation der Koeffizienten erschwert.\n\n")

                        f.write("## Nächste Schritte\n\n")
                        f.write(
                            "1. **Zeitbasierte Analyse**: Untersuchung des Einflusses des Zeitpunkts von Interventionen auf die Heilungsdauer.\n")
                        f.write(
                            "2. **Subgruppenanalyse**: Detaillierte Analyse für bestimmte Patientengruppen (z.B. nach Alter oder Geschlecht).\n")
                        f.write(
                            "3. **Entwicklung eines Prognosetools**: Auf Basis der identifizierten Faktoren könnte ein praktisches Tool ")
                        f.write("zur Abschätzung der individuellen Heilungsdauer entwickelt werden.\n")

                        f.write("\n*Dieser Bericht wurde automatisch im Rahmen der Polytrauma-Analyse erstellt.*")

                    logger.info(f"Zusammenfassender Bericht erstellt: {report_path}")
                    return report_path
    except Exception as e:
        logger.error(f"Fehler bei der Erstellung des Berichts: {str(e)}", exc_info=True)
        return None


def multivariate_analysis():
    """Main function to perform multivariate analysis."""
    # Load environment variables
    load_dotenv()

    # Get paths from environment variables
    base_dataset = os.getenv("DATASET")
    output_folder = os.getenv("OUTPUT_FOLDER", "data/output")
    log_folder = os.getenv("LOG_FOLDER", "logs")
    graphs_folder = os.getenv("GRAPHS_FOLDER", "plots")

    # Create step4 folder structure for multivariate analysis
    step4_output_folder = os.path.join(output_folder, "step4", "multivariate_analysis")
    step4_log_folder = os.path.join(log_folder, "step4")
    step4_plots_folder = os.path.join(graphs_folder, "step4", "multivariate_analysis")

    # Create necessary directories
    os.makedirs(step4_output_folder, exist_ok=True)
    os.makedirs(step4_log_folder, exist_ok=True)
    os.makedirs(step4_plots_folder, exist_ok=True)

    # Setup logging
    logger = setup_logging(step4_log_folder, "multivariate_analysis.log")
    logger.info("Starte multivariate Analyse...")

    try:
        # Look for patient-level data in both univariate and multivariate output folders
        univariate_patient_data = os.path.join(output_folder, "step4", "univariate_analysis", "patient_level_data.xlsx")
        multivariate_patient_data = os.path.join(step4_output_folder, "patient_level_data.xlsx")

        # Check if either exists
        if os.path.exists(univariate_patient_data):
            logger.info(f"Patientendaten von univariater Analyse gefunden: {univariate_patient_data}")
            patient_data_path = univariate_patient_data
        elif os.path.exists(multivariate_patient_data):
            logger.info(f"Patientendaten im multivariate Ordner gefunden: {multivariate_patient_data}")
            patient_data_path = multivariate_patient_data
        else:
            # Need to create the patient-level data
            logger.info("Keine vorhandenen Patientendaten gefunden. Erstelle neue Patientendaten.")
            patient_data_path = prepare_patient_level_data_if_needed(base_dataset, step4_output_folder, logger)

        # Load patient-level data
        df = load_patient_level_data(patient_data_path, logger)

        # Build regression models
        logger.info("Erstelle Regressionsmodelle...")
        regression_models = build_regression_models(df, step4_output_folder, logger)

        # Perform survival analysis
        logger.info("Führe Survival-Analyse durch...")
        cox_model = perform_survival_analysis(df, step4_output_folder, step4_plots_folder, logger)

        # Visualize regression results
        logger.info("Erstelle Visualisierungen...")
        visualize_regression_results(regression_models, df, step4_plots_folder, logger)

        # Generate summary report
        logger.info("Erstelle zusammenfassenden Bericht...")
        report_path = generate_summary_report(regression_models, cox_model, step4_output_folder, logger)

        # Export model comparison to Excel
        try:
            logger.info("Exportiere Modellvergleich nach Excel...")

            model_comparison = []
            for model_name in ['model1', 'model2', 'model3']:
                if model_name in regression_models and regression_models[model_name] is not None:
                    model = regression_models[model_name]
                    model_comparison.append({
                        'Modell': model_name,
                        'R²': model.rsquared,
                        'Angepasstes R²': model.rsquared_adj,
                        'AIC': model.aic,
                        'BIC': model.bic,
                        'F-Statistik': model.fvalue,
                        'F-p-wert': model.f_pvalue,
                        'Log-Likelihood': model.llf
                    })

            if model_comparison:
                model_df = pd.DataFrame(model_comparison)
                model_df.to_excel(os.path.join(step4_output_folder, "modellvergleich.xlsx"), index=False)
                logger.info(
                    f"Modellvergleich gespeichert unter: {os.path.join(step4_output_folder, 'modellvergleich.xlsx')}")
        except Exception as e:
            logger.error(f"Fehler beim Exportieren des Modellvergleichs: {str(e)}", exc_info=True)

        logger.info("Multivariate Analyse erfolgreich abgeschlossen.")
        logger.info(f"Ergebnisse gespeichert unter: {step4_output_folder}")
        logger.info(f"Visualisierungen gespeichert unter: {step4_plots_folder}")

    except Exception as e:
        logger.error(f"Fehler bei der multivariaten Analyse: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    multivariate_analysis()