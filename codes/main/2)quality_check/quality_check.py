import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
DATASET = os.getenv("DATASET")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
LOG_FOLDER = os.getenv("LOG_FOLDER")
GRAPHS_FOLDER = os.getenv("GRAPHS_FOLDER")

# Create step2 folders
step2_output_folder = os.path.join(OUTPUT_FOLDER, "step2")
step2_log_folder = os.path.join(LOG_FOLDER, "step2")
step2_plots_folder = os.path.join(GRAPHS_FOLDER, "step2")

# Create necessary directories
os.makedirs(step2_output_folder, exist_ok=True)
os.makedirs(step2_log_folder, exist_ok=True)
os.makedirs(step2_plots_folder, exist_ok=True)

# Setup logging
log_file = os.path.join(step2_log_folder, "quality_check.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting data quality check...")

# Load the processed data
try:
    df = pd.read_excel(DATASET, dtype={"Schadennummer": str})
    df.columns = df.columns.str.strip()
    logging.info(f"Loaded data from {DATASET}")
    logging.info(f"Data shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
except Exception as e:
    logging.error(f"Error loading data: {str(e)}")
    raise


# Define expected column types based on column names
def determine_expected_type(column_name):
    """Determine expected data type based on column name patterns"""
    if "datum" in column_name.lower():
        return "datetime"
    elif "nummer" in column_name.lower() or "Schadennummer" == column_name:
        return "string"
    elif column_name in ["Monat nach Unfall", "Vor Ort-Besuch Nummer"]:  # Removed "Alter in Dekaden"
        return "numeric"
    elif column_name in ["Time Interval", "Time_Interval", "Days_Since_Accident", "Age_At_Accident"]:
        return "numeric"
    elif column_name in ["Alter in Dekaden"]:  # Added this as categorical
        return "categorical_string"
    elif "RM" in column_name or "Somatisch" in column_name or "Taetigkeit" in column_name or "Tätigkeit" in column_name:
        return "categorical"
    elif column_name in ["Kopf", "Hals", "Thorax", "Abdomen", "Arm links", "Arm rechts",
                         "Wirbelsaeule", "Bein rechts", "Bein links", "Becken",
                         "Geschlecht", "Altersrentner", "Ehrenamt", "Zuverdienst",
                         "Organisation Pflege"]:
        return "categorical"
    elif "Personenbezogen" in column_name or "Umwelt" in column_name:
        return "categorical"
    elif column_name in ["Ort der Beratung"]:
        return "categorical_string"
    else:
        return "unknown"


# Check for missing values
def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100

    missing_data = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentage
    })

    missing_data = missing_data.sort_values('Missing_Count', ascending=False)

    # Log columns with missing values
    columns_with_missing = missing_data[missing_data['Missing_Count'] > 0]
    logging.info(f"Found {len(columns_with_missing)} columns with missing values")

    if len(columns_with_missing) > 0:
        logging.info("Columns with missing values:")
        for col, row in columns_with_missing.iterrows():
            logging.info(f"  {col}: {row['Missing_Count']} missing values ({row['Missing_Percentage']:.2f}%)")

    # Plot missing values
    plt.figure(figsize=(12, 8))
    columns_to_plot = missing_data[missing_data['Missing_Count'] > 0].head(20).index

    if len(columns_to_plot) > 0:
        sns.barplot(x=missing_data.loc[columns_to_plot, 'Missing_Percentage'],
                    y=columns_to_plot)
        plt.title('Percentage of Missing Values by Column (Top 20)')
        plt.xlabel('Missing Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(step2_plots_folder, "missing_values.png"))
        plt.close()
        logging.info(f"Missing values visualization saved to {os.path.join(step2_plots_folder, 'missing_values.png')}")
    else:
        logging.info("No columns with missing values to visualize")

    return missing_data


# Validate data types
def validate_data_types(df):
    """Validate if data types match expected types based on column names"""
    type_validation = []

    for column in df.columns:
        expected_type = determine_expected_type(column)
        current_type = str(df[column].dtype)

        # Check if current type matches expected type
        if expected_type == "datetime":
            is_correct = pd.api.types.is_datetime64_any_dtype(df[column])
        elif expected_type == "string":
            is_correct = pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column])
        elif expected_type == "numeric":
            is_correct = pd.api.types.is_numeric_dtype(df[column])
        elif expected_type == "categorical":
            # Check if categorical columns contain only Ja/Nein values (plus NaN/None)
            unique_vals = set(df[column].dropna().unique())
            is_correct = unique_vals.issubset({"Ja", "Nein"}) or len(unique_vals) <= 5
        elif expected_type == "categorical_string":
            # For categorical strings, we just check it's a string/object type
            is_correct = pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column])
        else:
            is_correct = True  # Unknown types are assumed correct

        type_validation.append({
            'Column': column,
            'Expected_Type': expected_type,
            'Current_Type': current_type,
            'Is_Correct': is_correct
        })

    type_df = pd.DataFrame(type_validation)

    # Log incorrect data types
    incorrect_types = type_df[~type_df['Is_Correct']]
    logging.info(f"Found {len(incorrect_types)} columns with incorrect data types")

    if len(incorrect_types) > 0:
        logging.info("Columns with incorrect data types:")
        for _, row in incorrect_types.iterrows():
            logging.info(f"  {row['Column']}: Expected {row['Expected_Type']}, got {row['Current_Type']}")

    return type_df


# Check for duplicates
def check_duplicates(df):
    """Check for duplicate rows and identify which Schadennummer has duplicates"""
    # Check for complete duplicate rows
    duplicate_mask = df.duplicated(keep=False)
    duplicates = df[duplicate_mask]
    duplicate_rows_count = len(duplicates)

    logging.info(f"Number of completely duplicate rows: {duplicate_rows_count}")

    if duplicate_rows_count > 0:
        # Identify which Schadennummer has duplicates
        duplicate_cases = duplicates['Schadennummer'].unique()
        for case in duplicate_cases:
            case_duplicates = duplicates[duplicates['Schadennummer'] == case]
            logging.info(f"Schadennummer {case} has {len(case_duplicates)} duplicate rows")

            # Log the duplicate rows for inspection
            duplicate_pairs = []
            for i in range(len(case_duplicates)):
                for j in range(i + 1, len(case_duplicates)):
                    row1 = case_duplicates.iloc[i]
                    row2 = case_duplicates.iloc[j]
                    if (row1 == row2).all():
                        duplicate_pairs.append((i, j))
                        logging.info(
                            f"  100% identical rows found at indices {case_duplicates.index[i]} and {case_duplicates.index[j]}")

            # Save the duplicates for this case to a separate file for inspection
            case_duplicates.to_excel(os.path.join(step2_output_folder, f"duplicates_case_{case}.xlsx"), index=True)
            logging.info(f"  Duplicates for case {case} saved for inspection")

    # Check for duplicate Schadennummer entries (expected due to multiple visits)
    unique_cases = df['Schadennummer'].nunique()
    total_cases = len(df)
    logging.info(f"Number of unique cases (Schadennummer): {unique_cases}")
    logging.info(f"Total number of records: {total_cases}")

    # Check for duplicate visits (same Schadennummer and Besuchsdatum)
    if 'Besuchsdatum' in df.columns:
        visit_duplicates = df.duplicated(subset=['Schadennummer', 'Besuchsdatum'], keep=False)
        duplicate_visits = df[visit_duplicates].sort_values(['Schadennummer', 'Besuchsdatum'])
        duplicate_visits_count = len(duplicate_visits)

        logging.info(
            f"Number of potential duplicate visits (same Schadennummer and Besuchsdatum): {duplicate_visits_count}")

        if duplicate_visits_count > 0:
            logging.info("Potential duplicate visits found (could be legitimate multiple visits on same day):")

            # Group by Schadennummer and Besuchsdatum to analyze each visit
            grouped_visits = duplicate_visits.groupby(['Schadennummer', 'Besuchsdatum'])
            for (case, date), group in grouped_visits:
                identical_rows = group.duplicated(keep=False)
                if identical_rows.all():  # All rows in this group are identical
                    logging.warning(f"  Case {case} on {date} has {len(group)} IDENTICAL visits - likely data error")
                else:
                    logging.info(f"  Case {case} on {date} has {len(group)} different visits - may be legitimate")

            # Save all duplicate visits to a file for inspection
            duplicate_visits.to_excel(os.path.join(step2_output_folder, "duplicate_visits.xlsx"), index=True)
            logging.info("  All duplicate visits saved for inspection")


# Validate categorical columns (checking if they contain only Ja/Nein values)
def validate_categorical_columns(df):
    """Verify if categorical columns contain only expected values"""
    categorical_issues = []

    # Define categories from your project structure
    categories = {
        "Körperteil": ['Kopf', 'Hals', 'Thorax', 'Abdomen', 'Arm links', 'Arm rechts', 'Wirbelsaeule', 'Bein rechts',
                       'Bein links', 'Becken'],
        "Somatisch": ['Somatisch-- Funktionsstoerung', 'Somatisch-- Schmerz', 'Somatisch--Komplikationen'],
        "Personenbezogen": ['Personenbezogen--Psychische Probleme/Compliance',
                            'Personenbezogen--Entschädigungsbegehren',
                            'Personenbezogen--Migrationshintergrund', 'Personenbezogen--Suchtverhalten',
                            'Personenbezogen--Zusätzliche Erkrankungen'],
        "Taetigkeit": ['Taetigkeit--Arbeitsunfähig', 'Tätigkeit--Wiedereingliederung', 'Tätigkeit--Arbeitsfaehig',
                       'Tätigkeit--BU/EU', 'Altersrentner', 'Ehrenamt', 'Zuverdienst'],
        "Umwelt": ['Umwelt--Beziehungsprobleme', 'Umwelt--Soziale Isolation', 'Umwelt--Mobilitaetsprobleme',
                   'Umwelt--Wohnsituatuation', 'Umwelt--Finazielle Probleme'],
        "Med RM": ['Med RM--Arzt-Vorstellung', 'Med RM--Arzt-Wechsel', 'Med RM--Organisation ambulante Therapie',
                   'Med RM--Organisation medizinische Reha', 'Med RM--Wietere Krankenhausaufenthalte',
                   'Med RM--Psychotherapie', 'Organisation Pflege'],
        "Soziales RM": ['Soziales RM--Lohnersatzleistungen', 'Soziales RM--Arbeitslosenunterstuetzung',
                        'Soziales RM--Antrag auf Sozialleistungen', 'Soziales RM--Einleitung Begutachtung'],
        "Technisches RM": ['Technisches RM--Hilfmittelversorgung', 'Technisches RM--Mobilitätshilfe',
                           'Technisches RM--Bauliche Anpassung', 'Technisches RM--Arbetsplatzanpassung'],
        "Berufliches RM": ['Berufliches RM--Leistungen zur Teilhabe am Arbeitsleben',
                           'Berufliches RM--Integration/berufliche Neurientierung allgemeiner Arbeitsmarkt',
                           'Berufliches RM--Wiedereingliederung geförderter Arbeitsmarkt', 'Berufliches RM--BEM']
    }

    # Flatten all category columns
    all_categorical_columns = []
    for category_cols in categories.values():
        all_categorical_columns.extend(category_cols)

    # Check each categorical column
    for column in all_categorical_columns:
        if column in df.columns:
            # Get unique non-null values
            unique_values = set(df[column].dropna().unique())
            # Check if values are as expected
            if not unique_values.issubset({"Ja", "Nein"}):
                unexpected_values = unique_values - {"Ja", "Nein"}
                categorical_issues.append({
                    'Column': column,
                    'Unexpected_Values': list(unexpected_values),
                    'Count': len(unexpected_values)
                })
                logging.warning(f"Column {column} contains unexpected values: {unexpected_values}")

    # Log summary
    if categorical_issues:
        logging.warning(f"Found {len(categorical_issues)} categorical columns with unexpected values")
    else:
        logging.info("All categorical columns contain only expected values (Ja/Nein)")

    return categorical_issues


# Replace missing values with "Nein" in categorical columns
def replace_missing_values(df):
    """Replace missing values in categorical columns with "Nein" and log changes"""
    # Define categories from your project structure (same as above)
    categories = {
        "Körperteil": ['Kopf', 'Hals', 'Thorax', 'Abdomen', 'Arm links', 'Arm rechts', 'Wirbelsaeule', 'Bein rechts',
                       'Bein links', 'Becken'],
        "Somatisch": ['Somatisch-- Funktionsstoerung', 'Somatisch-- Schmerz', 'Somatisch--Komplikationen'],
        "Personenbezogen": ['Personenbezogen--Psychische Probleme/Compliance',
                            'Personenbezogen--Entschädigungsbegehren',
                            'Personenbezogen--Migrationshintergrund', 'Personenbezogen--Suchtverhalten',
                            'Personenbezogen--Zusätzliche Erkrankungen'],
        "Taetigkeit": ['Taetigkeit--Arbeitsunfähig', 'Tätigkeit--Wiedereingliederung', 'Tätigkeit--Arbeitsfaehig',
                       'Tätigkeit--BU/EU', 'Altersrentner', 'Ehrenamt', 'Zuverdienst'],
        "Umwelt": ['Umwelt--Beziehungsprobleme', 'Umwelt--Soziale Isolation', 'Umwelt--Mobilitaetsprobleme',
                   'Umwelt--Wohnsituatuation', 'Umwelt--Finazielle Probleme'],
        "Med RM": ['Med RM--Arzt-Vorstellung', 'Med RM--Arzt-Wechsel', 'Med RM--Organisation ambulante Therapie',
                   'Med RM--Organisation medizinische Reha', 'Med RM--Wietere Krankenhausaufenthalte',
                   'Med RM--Psychotherapie', 'Organisation Pflege'],
        "Soziales RM": ['Soziales RM--Lohnersatzleistungen', 'Soziales RM--Arbeitslosenunterstuetzung',
                        'Soziales RM--Antrag auf Sozialleistungen', 'Soziales RM--Einleitung Begutachtung'],
        "Technisches RM": ['Technisches RM--Hilfmittelversorgung', 'Technisches RM--Mobilitätshilfe',
                           'Technisches RM--Bauliche Anpassung', 'Technisches RM--Arbetsplatzanpassung'],
        "Berufliches RM": ['Berufliches RM--Leistungen zur Teilhabe am Arbeitsleben',
                           'Berufliches RM--Integration/berufliche Neurientierung allgemeiner Arbeitsmarkt',
                           'Berufliches RM--Wiedereingliederung geförderter Arbeitsmarkt', 'Berufliches RM--BEM']
    }

    # Flatten all category columns
    all_categorical_columns = []
    for category_cols in categories.values():
        all_categorical_columns.extend(category_cols)

    # Count missing values before replacement
    missing_before = df[all_categorical_columns].isnull().sum().sum()

    # Copy the dataframe for modification
    df_clean = df.copy()

    # Replace NaN with "Nein" in categorical columns
    for column in all_categorical_columns:
        if column in df_clean.columns:
            missing_in_column = df_clean[column].isnull().sum()
            if missing_in_column > 0:
                df_clean[column] = df_clean[column].fillna("Nein")
                logging.info(f"Replaced {missing_in_column} missing values with 'Nein' in column {column}")

    # Count missing values after replacement
    missing_after = df_clean[all_categorical_columns].isnull().sum().sum()
    total_replaced = missing_before - missing_after

    logging.info(f"Total missing values in categorical columns before replacement: {missing_before}")
    logging.info(f"Total missing values in categorical columns after replacement: {missing_after}")
    logging.info(f"Total replaced values: {total_replaced}")

    return df_clean


# Remove exactly identical rows (100% match)
def remove_duplicate_rows(df):
    """
    Remove rows that are 100% identical (all columns match)
    """
    # Count duplicates before removal
    duplicates_before = df.duplicated().sum()

    if duplicates_before > 0:
        # Find the indices of duplicate rows
        duplicate_indices = df[df.duplicated()].index.tolist()
        logging.info(f"Found {duplicates_before} 100% identical duplicate rows at indices: {duplicate_indices}")

        # Save the duplicates for inspection before removal
        df[df.duplicated(keep=False)].to_excel(os.path.join(step2_output_folder, "identical_duplicates.xlsx"),
                                               index=True)

        # Remove duplicates (keep first occurrence)
        df_clean = df.drop_duplicates()

        logging.info(f"Removed {duplicates_before} 100% identical duplicate rows, keeping first occurrence")
        logging.info(f"Data shape after removing duplicates: {df_clean.shape}")

        return df_clean
    else:
        logging.info("No 100% identical duplicate rows found")
        return df


# Generate a summary report
def generate_summary(df, missing_data, type_validation, categorical_issues):
    """Generate a summary report of data quality"""
    summary = {
        "Total_Records": len(df),
        "Unique_Cases": df['Schadennummer'].nunique(),
        "Total_Columns": len(df.columns),
        "Columns_With_Missing": len(missing_data[missing_data['Missing_Count'] > 0]),
        "Columns_With_Type_Issues": len(type_validation[~type_validation['Is_Correct']]),
        "Columns_With_Categorical_Issues": len(categorical_issues),
        "Total_Missing_Values": missing_data['Missing_Count'].sum(),
        "Highest_Missing_Percentage": missing_data['Missing_Percentage'].max() if len(missing_data) > 0 else 0,
        "Duplicate_Rows": df.duplicated().sum()
    }

    # Log summary
    logging.info("Data Quality Summary:")
    for key, value in summary.items():
        logging.info(f"  {key}: {value}")

    # Create a summary dataframe
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(step2_output_folder, "data_quality_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    logging.info(f"Data quality summary saved to {summary_path}")

    return summary


def propagate_patient_level_data(df, logger):
    """
    Fill missing values in patient-level fields by propagating values within each Schadennummer.
    This assumes information like gender, birth date, etc. should be consistent across all visits
    for the same patient.
    """
    logger.info("Propagating patient-level data across visits...")

    # List of columns to propagate (patient-level data that should be consistent)
    patient_columns = [
        'Geschlecht',
        'Gebursdatum',
        'Alter in Dekaden',
        'Age_At_Accident'
    ]

    # Track changes
    changes_made = 0

    # Process each patient
    for patient_id in df['Schadennummer'].unique():
        patient_mask = df['Schadennummer'] == patient_id
        patient_data = df.loc[patient_mask]

        # For each patient-level column
        for column in patient_columns:
            if column not in df.columns:
                continue

            # Find non-null values for this patient in this column
            valid_values = patient_data[column].dropna()

            # If we have any valid values, propagate to all visits for this patient
            if len(valid_values) > 0:
                # Get the first non-null value
                value_to_use = valid_values.iloc[0]

                # Count current nulls for this patient in this column
                null_count = patient_data[column].isnull().sum()

                if null_count > 0:
                    # Fill nulls with the value
                    df.loc[patient_mask, column] = df.loc[patient_mask, column].fillna(value_to_use)
                    changes_made += null_count
                    logger.info(f"  Filled {null_count} missing values in {column} for patient {patient_id}")

    # Generate decades from Age_At_Accident where Alter in Dekaden is still missing
    if 'Age_At_Accident' in df.columns and 'Alter in Dekaden' in df.columns:
        # Find rows where Age_At_Accident exists but Alter in Dekaden is missing
        mask = df['Age_At_Accident'].notna() & df['Alter in Dekaden'].isna()

        if mask.any():
            # Calculate decade from age and format as string with "er" suffix
            decades = (df.loc[mask, 'Age_At_Accident'] // 10 * 10).astype(int).astype(str) + "er"
            df.loc[mask, 'Alter in Dekaden'] = decades
            decade_changes = mask.sum()
            changes_made += decade_changes
            logger.info(f"  Generated {decade_changes} decade values from Age_At_Accident")

    logger.info(f"Total patient-level values propagated: {changes_made}")

    return df


def main():
    try:
        # Use the global df variable
        global df

        # Perform quality checks
        logging.info("Analyzing missing values...")
        missing_data = analyze_missing_values(df)

        logging.info("Validating data types...")
        type_validation = validate_data_types(df)

        logging.info("Checking for duplicates...")
        check_duplicates(df)

        logging.info("Validating categorical columns...")
        categorical_issues = validate_categorical_columns(df)

        logging.info("Propagating patient-level data across visits...")
        df = propagate_patient_level_data(df, logging)  # Use logging instead of logger

        logging.info("Replacing missing values in categorical columns...")
        df_clean = replace_missing_values(df)

        logging.info("Removing 100% identical duplicate rows...")
        df_clean = remove_duplicate_rows(df_clean)

        # Generate summary
        logging.info("Generating quality summary...")
        generate_summary(df_clean, missing_data, type_validation, categorical_issues)

        # Save the cleaned dataset
        cleaned_output_path = os.path.join(step2_output_folder, "Polytrauma_Analysis_Clean.xlsx")
        df_clean.to_excel(cleaned_output_path, index=False)
        logging.info(f"Cleaned dataset saved to {cleaned_output_path}")

        # Save detailed quality check results
        missing_data.to_excel(os.path.join(step2_output_folder, "missing_values_analysis.xlsx"))
        type_validation.to_excel(os.path.join(step2_output_folder, "data_type_validation.xlsx"))
        if categorical_issues:
            pd.DataFrame(categorical_issues).to_excel(os.path.join(step2_output_folder, "categorical_issues.xlsx"))

        logging.info("Data quality check completed successfully")

    except Exception as e:
        logging.error(f"Error in data quality check: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()