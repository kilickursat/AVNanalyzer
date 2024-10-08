import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.interpolate import griddata
from scipy import stats
import matplotlib.pyplot as plt
import base64

# Set page config at the very beginning
st.set_page_config(
    page_title="Herrenknecht Hard Rock Data Analysis App",
    # Optional: Add a page icon if available
    # page_icon="URL_TO_YOUR_ICON"
)

# Helper function to clean numeric columns
def clean_numeric_column(df, column_name):
    df[column_name] = df[column_name].replace(r'[^0-9.-]+', '', regex=True)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df[column_name] = df[column_name].fillna(df[column_name].median())
    return df

# Function to calculate torque
def calculate_torque(working_pressure, torque_constant, current_speed=None, n1=None):
    """
    Calculate torque based on working pressure and motor speed conditions.
    Supports vectorized operations for Pandas Series.

    Parameters:
    - working_pressure (Series or float): Working pressure values.
    - torque_constant (float): Torque constant.
    - current_speed (Series or float, optional): Current speed values.
    - n1 (float, optional): Threshold speed.

    Returns:
    - Series or float: Calculated torque values.
    """
    if current_speed is None or n1 is None:
        # Without Variable Speed Motor
        torque = working_pressure * torque_constant
    else:
        # With Variable Speed Motor
        # Use numpy.where for vectorized conditional calculation
        torque = np.where(
            current_speed < n1,
            working_pressure * torque_constant,
            (n1 / current_speed) * torque_constant * working_pressure
        )

    return torque

# Function to calculate derived features
def calculate_derived_features(df, working_pressure_col, advance_rate_col, revolution_col, n1, torque_constant):
    """
    Calculate derived features: "Calculated torque [kNm]" and "Penetration Rate [mm/rev]".

    Parameters:
    - df (DataFrame): Input dataframe.
    - working_pressure_col (str): Column name for working pressure.
    - advance_rate_col (str): Column name for advance rate.
    - revolution_col (str): Column name for revolution.
    - n1 (float): Threshold speed.
    - torque_constant (float): Torque constant.

    Returns:
    - DataFrame: DataFrame with new derived features.
    """
    try:
        # Calculate "Calculated torque [kNm]"
        if working_pressure_col != 'None':
            df["Calculated torque [kNm]"] = calculate_torque(
                working_pressure=df[working_pressure_col],
                torque_constant=torque_constant,
                current_speed=df[advance_rate_col] if advance_rate_col != 'None' else None,
                n1=n1
            )
        else:
            df["Calculated torque [kNm]"] = np.nan

        # Calculate "Penetration Rate [mm/rev]" as Advance Rate / Revolution
        if advance_rate_col != 'None' and revolution_col != 'None':
            df["Penetration Rate [mm/rev]"] = df[advance_rate_col] / df[revolution_col]
        else:
            df["Penetration Rate [mm/rev]"] = np.nan  # Assign NaN if columns are not selected

        return df
    except Exception as e:
        st.error(f"Error calculating derived features: {e}")
        return df

# Enhanced Function to read CSV or Excel file with validation
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None

        if df.empty:
            st.error("The uploaded file is empty or not formatted correctly.")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def read_rock_strength_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported rock strength file format. Please upload a CSV or Excel file.")
            return None

        if df.empty:
            st.error("The uploaded rock strength file is empty or not formatted correctly.")
            return None

        return df
    except Exception as e:
        st.error(f"An error occurred while reading rock strength data: {e}")
        return None

# Function to preprocess rock strength data
def preprocess_rock_strength_data(df):
    try:
        df['Rock Type'] = df['Probenbezeichnung'].str.split().str[0]
        pivoted = df.pivot_table(values='Value', index='Rock Type', columns='Test', aggfunc='mean')
        pivoted.rename(columns={'UCS': 'UCS (MPa)', 'BTS': 'BTS (MPa)', 'PLT': 'PLT (MPa)'}, inplace=True)
        return pivoted
    except Exception as e:
        st.error(f"Error preprocessing rock strength data: {e}")
        return None

# Function to suggest column based on keywords
def suggest_column(df, keywords):
    for keyword in keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
    return None

# Function to create correlation heatmap
def create_correlation_heatmap(df, selected_features):
    try:
        corr = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {e}")

# Function to create statistical summary
def create_statistical_summary(df, selected_features):
    try:
        summary = df[selected_features].describe()
        st.table(summary)
    except Exception as e:
        st.error(f"Error creating statistical summary: {e}")

# Function to create features vs time plot
def create_features_vs_time(df, selected_features, time_column):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        for feature in selected_features:
            ax.plot(df[time_column], df[feature], label=feature)
        ax.set_xlabel(time_column)
        ax.set_ylabel("Value")
        ax.set_title("Features vs Time")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating features vs time plot: {e}")

# Function to create pressure distribution polar plot
def create_pressure_distribution_polar_plot(df, working_pressure_col, time_column):
    try:
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=df[working_pressure_col],
            theta=df[time_column],
            mode='lines',
            name='Working Pressure [bar]'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    title="Working Pressure [bar]"
                )
            ),
            showlegend=True,
            title="Pressure Distribution"
        )

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating pressure distribution polar plot: {e}")

# Function to create parameters vs chainage plot
def create_parameters_vs_chainage(df, selected_features, selected_distance):
    try:
        fig = make_subplots(rows=1, cols=1)

        for feature in selected_features:
            fig.add_trace(go.Scatter(
                x=df[selected_distance],
                y=df[feature],
                mode='lines+markers',
                name=feature
            ))

        fig.update_layout(
            title=f"Parameters vs {selected_distance}",
            xaxis_title=selected_distance,
            yaxis_title="Value",
            height=600
        )

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating parameters vs chainage plot: {e}")

# Function to create box plots
def create_multi_axis_box_plots(df, selected_features):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[selected_features], ax=ax)
        ax.set_title("Box Plots")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating box plots: {e}")

# Function to create violin plots
def create_multi_axis_violin_plots(df, selected_features):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df[selected_features], ax=ax)
        ax.set_title("Violin Plots")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating violin plots: {e}")

# Function to create thrust force plots
def create_thrust_force_plots(df):
    try:
        thrust_force_col = suggest_column(df, ['thrust force', 'vorschubkraft', 'kraft','kraft_max','gesamtkraft','gesamtkraft_stz','gesamtkraft_vtp'])
        penetration_rate_col = 'Penetration Rate [mm/rev]'  # As per derived feature
        advance_rate_col = 'Advance Rate [mm/min]'

        if thrust_force_col is None:
            st.warning("Thrust force column not found in the dataset.")
            return

        if penetration_rate_col not in df.columns:
            st.warning("Penetration Rate [mm/rev] column not found. Ensure it is calculated correctly.")
            return

        if advance_rate_col not in df.columns:
            st.warning("Advance Rate [mm/min] column not found.")
            return

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Thrust Force vs Penetration Rate", "Thrust Force vs Advance Rate"))

        # Thrust Force vs Penetration Rate
        fig.add_trace(go.Scatter(
            x=df[penetration_rate_col], 
            y=df[thrust_force_col], 
            mode='markers', 
            name='Penetration Rate [mm/rev]', 
            marker=dict(color='blue', size=5)
        ), row=1, col=1)

        # Thrust Force vs Advance Rate
        fig.add_trace(go.Scatter(
            x=df[advance_rate_col], 
            y=df[thrust_force_col], 
            mode='markers', 
            name='Advance Rate [mm/min]',
            marker=dict(color='green', size=5)
        ), row=2, col=1)

        fig.update_layout(
            height=800, 
            width=800, 
            title_text="Thrust Force Relationships"
        )
        fig.update_xaxes(title_text="Penetration Rate [mm/rev]", row=1, col=1)
        fig.update_xaxes(title_text="Advance Rate [mm/min]", row=2, col=1)
        fig.update_yaxes(title_text="Thrust Force [kN]", row=1, col=1)
        fig.update_yaxes(title_text="Thrust Force [kN]", row=2, col=1)

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating thrust force plots: {e}")

# Function to create rock strength comparison chart
def create_rock_strength_comparison_chart(machine_df, rock_df, rock_type, selected_features):
    try:
        # Filter rock_df for the selected rock_type
        rock_df = rock_df[rock_df.index == rock_type]
        if rock_df.empty:
            st.error(f"No data available for {rock_type} rock type.")
            return

        # Calculate average values of selected machine features
        avg_values = machine_df[selected_features].mean()

        parameters = []
        machine_values = []
        for feature in selected_features:
            if any(keyword in feature.lower() for keyword in ['advance rate', 'vortrieb', 'vorschub', 'vtsgeschw', 'geschw']):
                parameters.append('Advance rate [mm/min]')
            elif any(keyword in feature.lower() for keyword in ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'sr_drehz']):
                parameters.append('Revolution [rpm]')
            elif any(keyword in feature.lower() for keyword in ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'sr_arbdr']):
                parameters.append('Working pressure [bar]')
            else:
                parameters.append(feature)
            machine_values.append(avg_values[feature])

        # Corrected Line: Ensure proper closing parenthesis
        ucs_values = [rock_df['UCS (MPa)'].iloc[0]] * len(selected_features)
        bts_values = [rock_df['BTS (MPa)'].iloc[0]] * len(selected_features)  # Fixed here
        plt_values = [rock_df['PLT (MPa)'].iloc[0]] * len(selected_features)

        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=100)
        fig.suptitle(f"Machine Parameters vs {rock_type} Rock Strength", fontsize=20, fontweight='bold')

        colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5']

        for i, (param, ax) in enumerate(zip(parameters, axs.flat)):
            x = np.arange(4)
            width = 0.2

            bars = ax.bar(x, [machine_values[i], ucs_values[i], bts_values[i], plt_values[i]], width, color=colors, edgecolor='black')

            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.set_title(f'{rock_type} - {param}', fontsize=16, fontweight='bold')
            ax.set_xticks(x)

            ax.set_xticklabels(['Machine', 'UCS', 'BTS', 'PLT'], fontsize=12, fontweight='bold')

            ax.tick_params(axis='y', labelsize=10)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error creating rock strength comparison chart: {e}")

# Function to add logo to sidebar
def add_logo():
    try:
        st.sidebar.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                background-image: url(https://github.com/kilickursat/AVNanalyzer/raw/main/Herrenknecht_logo.svg);
                background-repeat: no-repeat;
                background-size: 120px;
                background-position: 10px 10px;
                padding-top: 120px;  /* Reduced padding */
            }
            [data-testid="stSidebar"]::before {
                content: "";
                margin-bottom: 20px;  /* Reduced margin */
                display: block;
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 0;  /* Remove additional padding */
            }
            .sidebar-content {
                padding-top: 0;  /* Remove additional padding */
            }
            .sidebar-content > * {
                margin-bottom: 0.5rem !important;
            }
            /* Reduce the size of the headers in the sidebar */
            .sidebar .sidebar-content div[data-testid="stMarkdownContainer"] > h1 {
                font-size: 1.5em;
                margin-top: 0;
            }
            .sidebar .sidebar-content div[data-testid="stMarkdownContainer"] > h2 {
                font-size: 1.2em;
                margin-top: 0;
            }
            /* Make the file uploader more compact */
            .sidebar .sidebar-content [data-testid="stFileUploader"] {
                margin-bottom: 0.5rem;
            }
            /* Adjust radio button spacing */
            .sidebar .sidebar-content [data-testid="stRadio"] {
                margin-bottom: 0.5rem;
            }
            .sidebar .sidebar-content [data-testid="stRadio"] > div {
                margin-bottom: 0.2rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error adding logo: {e}")

# Safe selectbox function
def safe_selectbox(label, options, suggested_option):
    """
    Safely create a selectbox, defaulting to 'None' if suggested_option is not in options.

    Parameters:
    - label (str): Label for the selectbox.
    - options (list): List of options.
    - suggested_option (str): Suggested default option.

    Returns:
    - str: Selected option.
    """
    try:
        if suggested_option and suggested_option in options:
            index = options.index(suggested_option)
        else:
            index = 0  # Default to 'None'
    except ValueError:
        index = 0  # Default to 'None' if suggested_option is not in options
    return st.sidebar.selectbox(label, options, index=index)

# Main function
def main():
    try:
        # Add logo to sidebar
        add_logo()

        st.title("Herrenknecht Hard Rock Data Analysis App")

        # Sidebar for file upload and visualization selection
        st.sidebar.header("Data Upload & Analysis")

        # File upload
        uploaded_file = st.sidebar.file_uploader("Machine Data (CSV/Excel)", type=['csv', 'xlsx'])
        rock_strength_file = st.sidebar.file_uploader("Rock Strength Data (CSV/Excel)", type=['csv', 'xlsx'])

        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.success("Machine data loaded successfully.")

                # Suggest columns based on keywords
                suggested_working_pressure = suggest_column(df, ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'sr_arbdr'])
                suggested_revolution = suggest_column(df, ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'sr_drehz'])
                suggested_advance_rate = suggest_column(df, ['advance rate', 'vortrieb', 'vorschub', 'VTgeschw', 'geschw'])

                # Select columns with safe_selectbox
                working_pressure_cols = df.columns.tolist()
                revolution_cols = df.columns.tolist()
                advance_rate_cols = df.columns.tolist()

                working_pressure_col = safe_selectbox(
                    "Select Working Pressure Column", 
                    ['None'] + working_pressure_cols,
                    suggested_working_pressure
                )
                revolution_col = safe_selectbox(
                    "Select Revolution Column", 
                    ['None'] + revolution_cols,
                    suggested_revolution
                )
                advance_rate_col = safe_selectbox(
                    "Select Advance Rate Column", 
                    ['None'] + advance_rate_cols,
                    suggested_advance_rate
                )

                # Set constants (assuming n1 and torque_constant are predefined)
                n1 = 100.0  # Example value, adjust as needed
                torque_constant = 0.5  # Example value in kNm/bar, adjust as needed

                # Calculate derived features
                df = calculate_derived_features(
                    df=df,
                    working_pressure_col=working_pressure_col,
                    advance_rate_col=advance_rate_col,
                    revolution_col=revolution_col,
                    n1=n1,
                    torque_constant=torque_constant
                )

                # Handle missing values if necessary
                df.fillna(method='ffill', inplace=True)

                # Visualization selection
                options = ['Correlation Heatmap', 'Statistical Summary', 'Features vs Time', 'Pressure Distribution', 
                           'Parameters vs Chainage', 'Box Plots', 'Violin Plots', 'Thrust Force Plots']
                
                if rock_strength_file:
                    options.append('Rock Strength Comparison')

                selected_option = st.sidebar.radio("Choose visualization", options)

                # Rock strength data processing
                rock_df = None
                rock_type = None
                if rock_strength_file:
                    rock_strength_data = read_rock_strength_data(rock_strength_file)
                    if rock_strength_data is not None:
                        rock_df = preprocess_rock_strength_data(rock_strength_data)
                        if rock_df is not None and not rock_df.empty:
                            rock_type = st.sidebar.selectbox("Select Rock Type", rock_df.index.tolist())
                        else:
                            st.warning("Rock strength data is empty after preprocessing.")

                # Main content area - Visualization based on user selection
                st.subheader(f"Visualization: {selected_option}")

                # Extract distance columns (assuming a helper function exists)
                distance_cols = [col for col in df.columns if 'distance' in col.lower() or 'chainage' in col.lower()]
                if distance_cols:
                    selected_distance = st.sidebar.selectbox("Select distance/chainage column", distance_cols)
                else:
                    selected_distance = 'None'

                # Let user select features for analysis
                selected_features = st.sidebar.multiselect("Select features for analysis", df.columns)

                # Check for time-related columns (assuming a helper function exists)
                time_columns = [col for col in df.columns if 'time' in col.lower()]
                if time_columns:
                    time_column = st.sidebar.selectbox("Select Time Column", time_columns)
                else:
                    time_column = 'None'

                if not selected_features and selected_option not in ['Pressure Distribution', 'Thrust Force Plots']:
                    st.warning("Please select at least one feature for analysis.")
                else:
                    if selected_option == 'Correlation Heatmap':
                        create_correlation_heatmap(df, selected_features)
                    elif selected_option == 'Statistical Summary':
                        create_statistical_summary(df, selected_features)
                    elif selected_option == 'Features vs Time' and time_column != 'None':
                        create_features_vs_time(df, selected_features, time_column)
                    elif selected_option == 'Pressure Distribution' and time_column != 'None':
                        if working_pressure_col != 'None':
                            create_pressure_distribution_polar_plot(df, working_pressure_col, time_column)
                        else:
                            st.warning("Please select a valid working pressure column.")
                    elif selected_option == 'Parameters vs Chainage':
                        if selected_distance != 'None':
                            create_parameters_vs_chainage(df, selected_features, selected_distance)
                        else:
                            st.warning("Please select a valid distance/chainage column.")
                    elif selected_option == 'Box Plots':
                        create_multi_axis_box_plots(df, selected_features)
                    elif selected_option == 'Violin Plots':
                        create_multi_axis_violin_plots(df, selected_features)
                    elif selected_option == 'Rock Strength Comparison':
                        if rock_df is not None and rock_type is not None:
                            create_rock_strength_comparison_chart(df, rock_df, rock_type, selected_features)
                        else:
                            st.warning("Please upload rock strength data and select a rock type to view the comparison.")
                    elif selected_option == 'Thrust Force Plots':
                        create_thrust_force_plots(df)

                # Add download button for processed data
                if st.sidebar.button("Download Processed Data"):
                    try:
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed CSV File</a>'
                        st.sidebar.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error preparing data for download: {e}")

            else:
                st.error("Error loading the data. Please check your file format.")
        else:
            st.info("Please upload a machine data file to begin analysis.")

        # Add footer
        st.markdown("---")
        st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
        st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

    except Exception as e:
        st.error(f"An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
