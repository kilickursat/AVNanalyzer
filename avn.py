import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Set page config at the very beginning
st.set_page_config(
    page_title="Herrenknecht Hard Rock Data Analysis App",
)

# Helper function to clean numeric columns
def clean_numeric_column(df, column_name):
    try:
        st.write(f"Cleaning and converting column: {column_name}")
        df[column_name] = df[column_name].astype(str)  # Ensure all data is string for regex
        df[column_name] = df[column_name].replace(r'[^0-9.-]+', '', regex=True)
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        median_value = df[column_name].median()
        st.write(f"Median value for {column_name}: {median_value}")
        df[column_name] = df[column_name].fillna(median_value)
        st.write(f"Converted {column_name} to numeric with median imputation.")
        return df
    except Exception as e:
        st.error(f"Error cleaning column {column_name}: {e}")
        return df

# Advanced rate calculation function
def calculate_advance_rate_and_stats(df, distance_column, time_column):
    try:
        if not all(col in df.columns for col in [distance_column, time_column]):
            raise ValueError(f"Required columns '{distance_column}' and/or '{time_column}' not found in DataFrame")
        
        st.write(f"Calculating advance rate using '{distance_column}' and '{time_column}'")

        # Verify data types
        st.write(f"Data type of '{distance_column}': {df[distance_column].dtype}")
        st.write(f"Data type of '{time_column}': {df[time_column].dtype}")

        if not pd.api.types.is_numeric_dtype(df[distance_column]):
            st.error(f"Distance column '{distance_column}' is not numeric.")
            return None, 0

        if not pd.api.types.is_numeric_dtype(df[time_column]):
            st.error(f"Time column '{time_column}' is not numeric.")
            return None, 0

        if len(df) > 1:
            weg = round(df[distance_column].max() - df[distance_column].min(), 2)
            zeit = round(df[time_column].max() - df[time_column].min(), 2)
        else:
            weg = round(df[distance_column].iloc[0], 2)
            zeit = round(df[time_column].iloc[0], 2)
        
        st.write(f"Total Distance (mm): {weg}")
        st.write(f"Total Time (seconds): {zeit}")

        # Assuming 'zeit' is in seconds; convert to minutes
        zeit_minutes = zeit / 60
        st.write(f"Total Time (min): {zeit_minutes}")

        average_speed = round(weg / zeit_minutes, 2) if zeit_minutes != 0 else 0
        st.write(f"Average Speed (mm/min): {average_speed}")
        
        result = {
            "Total Distance (mm)": weg,
            "Total Time (min)": zeit_minutes,
            "Average Speed (mm/min)": average_speed
        }
        
        return result, average_speed
        
    except Exception as e:
        st.error(f"Error in advance rate calculation: {e}")
        return None, 0

# Penetration rate calculation function
def calculate_penetration_rate(row, revolution_col):
    try:
        speed = row['Average Speed (mm/min)']
        revolution = row[revolution_col]
        
        if pd.isna(speed) or pd.isna(revolution):
            return np.nan
        elif revolution == 0:
            return np.inf if speed != 0 else 0
        else:
            return round(speed / revolution, 4)
            
    except Exception as e:
        st.error(f"Error in penetration rate calculation: {e}")
        return np.nan

# Function to calculate torque
def calculate_torque(working_pressure, torque_constant, current_speed=None, n1=None):
    if current_speed is None or n1 is None:
        torque = working_pressure * torque_constant
    else:
        if current_speed < n1:
            torque = working_pressure * torque_constant
        else:
            torque = (n1 / current_speed) * torque_constant * working_pressure
    return torque

# Function to calculate derived features
def calculate_derived_features(df, working_pressure_col, revolution_col, n1, torque_constant, selected_distance):
    try:
        if (working_pressure_col and working_pressure_col != 'None') and (revolution_col and revolution_col != 'None'):
            st.write("Calculating derived features...")

            # Clean working_pressure and revolution columns
            df = clean_numeric_column(df, working_pressure_col)
            df = clean_numeric_column(df, revolution_col)

            # Calculate torque
            df['Calculated torque [kNm]'] = df.apply(
                lambda row: calculate_torque(
                    row[working_pressure_col],
                    torque_constant,
                    current_speed=row[revolution_col],
                    n1=n1
                ),
                axis=1
            )
            st.write("Calculated 'Calculated torque [kNm]'")

            # Calculate average speed and penetration rate
            distance_column = selected_distance
            time_column = get_time_column(df)
            if distance_column and time_column:
                # Clean distance and time columns
                df = clean_numeric_column(df, distance_column)
                df = clean_numeric_column(df, time_column)

                result, average_speed = calculate_advance_rate_and_stats(df, distance_column, time_column)
                if result:
                    df['Average Speed (mm/min)'] = average_speed
                    st.write("Calculated 'Average Speed (mm/min)'")

                    if revolution_col and revolution_col != 'None':
                        df['Penetration Rate [mm/rev]'] = df.apply(
                            lambda row: calculate_penetration_rate(row, revolution_col),
                            axis=1
                        )
                        st.write("Calculated 'Penetration Rate [mm/rev]'")
        else:
            st.warning("Please select both Working Pressure and Revolution columns to calculate derived features.")
        return df
    except Exception as e:
        st.error(f"Error calculating derived features: {str(e)}")
        return df

# Helper functions for column identification
def identify_special_columns(df):
    working_pressure_keywords = ['working pressure', 'arbeitsdruck', 'sr_arbdr', 'sr_arbdr_z', 'pressure', 'druck', 'arbdr']
    revolution_keywords = ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'sr_drehz_z']
    advance_rate_keywords = ['advance rate', 'vorschub', 'penetration rate', 'vtgeschw', 'geschw', 'geschw_z']

    working_pressure_cols = [col for col in df.columns if any(kw in col.lower() for kw in working_pressure_keywords)]
    revolution_cols = [col for col in df.columns if any(kw in col.lower() for kw in revolution_keywords)]
    advance_rate_cols = [col for col in df.columns if any(kw in col.lower() for kw in advance_rate_keywords)]

    return working_pressure_cols, revolution_cols, advance_rate_cols

def get_distance_columns(df):
    distance_keywords = ['distance', 'length', 'travel', 'chainage', 'tunnellänge', 'weg_mm_z', 'vtp_weg']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in distance_keywords)]

def get_time_column(df):
    time_keywords = ['relativzeit', 'relative time', 'time', 'datum', 'date', 'zeit', 'timestamp']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            return col
    return None

# Enhanced Function to read CSV or Excel file with validation
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True)
            st.write("Loaded CSV file successfully.")
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
            st.write("Loaded Excel file successfully.")
        else:
            st.error("Unsupported file format")
            return None

        if df.empty:
            st.error("The uploaded file is empty or not formatted correctly.")
            return None

        st.write("Initial DataFrame:")
        st.write(df.head())
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def read_rock_strength_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True)
            st.write("Loaded Rock Strength CSV file successfully.")
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
            st.write("Loaded Rock Strength Excel file successfully.")
        else:
            st.error("Unsupported file format for Rock Strength data.")
            return None

        st.write("Rock Strength DataFrame:")
        st.write(df.head())
        return df
    except Exception as e:
        st.error(f"Error reading rock strength data: {e}")
        return None

# Function to preprocess the rock strength data
def preprocess_rock_strength_data(df):
    try:
        st.write("Preprocessing rock strength data...")
        df['Rock Type'] = df['Probenbezeichnung'].str.split().str[0]
        pivoted = df.pivot_table(values='Value', index='Rock Type', columns='Test', aggfunc='mean')
        pivoted.rename(columns={'UCS': 'UCS (MPa)', 'BTS': 'BTS (MPa)', 'PLT': 'PLT (MPa)'}, inplace=True)
        st.write("Preprocessed Rock Strength DataFrame:")
        st.write(pivoted.head())
        return pivoted
    except Exception as e:
        st.error(f"Error preprocessing rock strength data: {e}")
        return None

# Updated function to create comparison chart for machine parameters vs rock strength
def create_rock_strength_comparison_chart(machine_df, rock_df, rock_type, selected_features):
    try:
        st.write(f"Creating Rock Strength Comparison Chart for {rock_type}...")
        rock_df = rock_df[rock_df.index == rock_type]
        if rock_df.empty:
            st.error(f"No data available for {rock_type} rock type.")
            return

        avg_values = machine_df[selected_features].mean()
        st.write("Average values of selected features:")
        st.write(avg_values)

        parameters = []
        machine_values = []
        for feature in selected_features:
            if any(keyword in feature.lower() for keyword in ['advance rate', 'vortrieb', 'vorschub', 'vtgeschw', 'geschw']):
                parameters.append('Advance rate [mm/min]')
            elif any(keyword in feature.lower() for keyword in ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'sr_drehz_z']):
                parameters.append('Revolution [rpm]')
            elif any(keyword in feature.lower() for keyword in ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr_z', 'sr_arbdr']):
                parameters.append('Working pressure [bar]')
            else:
                parameters.append(feature)
            machine_values.append(avg_values[feature])

        ucs_values = [rock_df['UCS (MPa)'].iloc[0]] * len(selected_features)
        bts_values = [rock_df['BTS (MPa)'].iloc[0]] * len(selected_features)
        plt_values = [rock_df['PLT (MPa)'].iloc[0]] * len(selected_features)

        fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=100)  # Adjusted layout
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

# Updated function to create correlation heatmap with dynamic input
def create_correlation_heatmap(df, selected_features):
    try:
        st.write("Creating Correlation Heatmap...")
        if len(selected_features) < 2:
            st.warning("Please select at least two features for the correlation heatmap.")
            return

        # Only use the features that the user has explicitly selected
        available_features = [f for f in selected_features if f in df.columns]
        
        if len(available_features) < 2:
            st.warning("Please select at least two valid features for the correlation heatmap.")
            return

        corr_matrix = df[available_features].corr()
        st.write("Correlation Matrix:")
        st.write(corr_matrix)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title('Correlation Heatmap of Selected Parameters')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")

# Updated function to create statistical summary
def create_statistical_summary(df, selected_features, round_to=2):
    try:
        st.write("Creating Statistical Summary...")
        if not selected_features:
            st.warning("Please select at least one feature for the statistical summary.")
            return

        summary_dict = {}
        for feature in selected_features:
            summary_dict[feature] = {
                'count': int(df[feature].count()),
                'mean': round(df[feature].mean(), round_to),
                'median': round(df[feature].median(), round_to),
                'std': round(df[feature].std(), round_to),
                'min': round(df[feature].min(), round_to),
                '25%': round(df[feature].quantile(0.25), round_to),
                '50%': round(df[feature].quantile(0.50), round_to),
                '75%': round(df[feature].quantile(0.75), round_to),
                'max': round(df[feature].max(), round_to),
                'skewness': round(df[feature].skew(), round_to),
                'kurtosis': round(df[feature].kurtosis(), round_to)
            }

        summary = pd.DataFrame(summary_dict).transpose()
        st.write("Statistical Summary DataFrame:")
        st.write(summary)

        # Style the table (keeping the existing styling)
        styled_summary = summary.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': 'rgb(0, 62, 37)'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', 'rgb(0, 62, 37)'), ('color', 'white')]},
            {'selector': 'tbody tr:nth-of-type(even)', 'props': [('background-color', 'rgba(0, 62, 37, 0.1)')]},
            {'selector': 'tbody tr:last-of-type', 'props': [('border-bottom', '2px solid rgb(0, 62, 37)')]}
        ]).format(precision=round_to)

        # Convert styled DataFrame to HTML and remove the style block
        styled_html = styled_summary.to_html()
        styled_html = styled_html.split('</style>')[-1]  # Remove everything before and including </style>

        # Add custom CSS to ensure the table fits within the Streamlit container
        custom_css = """
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: right;
            padding: 8px;
            border: 1px solid rgb(0, 62, 37);
        }
        th {
            background-color: rgb(0, 62, 37);
            color: white;
        }
        tr:nth-of-type(even) {
            background-color: rgba(0, 62, 37, 0.1);
        }
        tbody tr:last-of-type {
            border-bottom: 2px solid rgb(0, 62, 37);
        }
        </style>
        """

        # Combine custom CSS with styled HTML table
        final_html = custom_css + styled_html

        # Display the styled table
        st.markdown(final_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error creating statistical summary: {str(e)}")

# Updated function to create features vs time
def create_features_vs_time(df, selected_features, time_column):
    try:
        st.write("Creating Features vs Time plot...")
        if not selected_features:
            st.warning("Please select at least one feature for the time series plot.")
            return

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5',
                  '#9B6B6B', '#E9967A', '#4682B4', '#6B8E23']  # Expanded color palette

        fig = make_subplots(rows=len(selected_features), cols=1,
                            shared_xaxes=True,
                            subplot_titles=selected_features,
                            vertical_spacing=0.05)  # Reduce spacing between subplots

        for i, feature in enumerate(selected_features, start=1):
            fig.add_trace(
                go.Scatter(
                    x=df[time_column],
                    y=df[feature],
                    mode='lines',
                    name=feature,
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=i,
                col=1
            )

            # Update y-axis titles
            fig.update_yaxes(title_text=feature, row=i, col=1)

        # Update layout with larger dimensions and better spacing
        fig.update_layout(
            height=400 * len(selected_features),  # Increased height per subplot
            width=1200,  # Increased overall width
            title_text='Features vs Time',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100, l=100, r=50, b=50)  # Adjusted margins
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating features vs time plot: {e}")

# Updated function to create pressure distribution polar plot with fixes
def create_pressure_distribution_polar_plot(df, pressure_column, time_column):
    try:
        st.write("Creating Pressure Distribution Polar Plot...")
        if pressure_column not in df.columns:
            st.error(f"Pressure column '{pressure_column}' not found in the dataset.")
            return

        # Ensure pressure_column is numeric
        df[pressure_column] = pd.to_numeric(df[pressure_column], errors='coerce')
        if df[pressure_column].isnull().all():
            st.error(f"All values in pressure column '{pressure_column}' are NaN after conversion.")
            return

        # Drop rows with NaN in pressure_column or time_column
        df_clean = df.dropna(subset=[pressure_column, time_column])
        if df_clean.empty:
            st.error("No valid data available for Pressure Distribution Polar Plot after cleaning.")
            return

        # Normalize time to 360 degrees
        df_clean['normalized_time'] = (df_clean[time_column] - df_clean[time_column].min()) / (df_clean[time_column].max() - df_clean[time_column].min()) * 360

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=df_clean[pressure_column],
            theta=df_clean['normalized_time'],
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Pressure'
        ))

        max_pressure = df_clean[pressure_column].max()
        if pd.isna(max_pressure) or max_pressure == 0:
            max_pressure = 1

        fig.update_layout(
            title='Pressure Distribution Over Time (Polar Plot)',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_pressure * 1.1],
                    showline=False,
                    showgrid=True,
                    gridcolor='lightgrey',
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[0, 90, 180, 270],
                    ticktext=['0°', '90°', '180°', '270°'],
                    direction='clockwise',
                    rotation=90
                )
            ),
            showlegend=False,
            template='plotly_white',
            height=600,
            width=600
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating pressure distribution polar plot: {e}")

# Function to rename columns for visualization
def rename_columns(df, working_pressure_col, revolution_col, distance_col, advance_rate_col):
    try:
        st.write("Renaming columns for visualization...")
        column_mapping = {}
        if working_pressure_col and working_pressure_col != 'None':
            column_mapping[working_pressure_col] = 'Working pressure [bar]'
        if revolution_col and revolution_col != 'None':
            column_mapping[revolution_col] = 'Revolution [rpm]'
        if distance_col:
            column_mapping[distance_col] = 'Chainage [mm]'
        if advance_rate_col and advance_rate_col != 'None':
            column_mapping[advance_rate_col] = 'Advance rate [mm/min]'
        
        df = df.rename(columns=column_mapping)
        st.write("Renamed Columns:")
        st.write(df.head())
        return df
    except Exception as e:
        st.error(f"Error renaming columns: {e}")
        return df

# Updated function to create Parameters vs Chainage plot with fixes
def create_parameters_vs_chainage(df, selected_features, chainage_column):
    try:
        st.write("Creating Parameters vs Chainage plot...")
        if not selected_features:
            st.warning("Please select at least one feature for the chainage plot.")
            return

        # Ensure the chainage column exists and is numeric
        if chainage_column not in df.columns:
            st.error(f"Chainage column '{chainage_column}' not found in the dataset.")
            return

        df[chainage_column] = pd.to_numeric(df[chainage_column], errors='coerce')
        if df[chainage_column].isnull().all():
            st.error(f"All values in chainage column '{chainage_column}' are NaN after conversion.")
            return

        # Drop rows with NaN in chainage or selected features
        df_clean = df.dropna(subset=[chainage_column] + selected_features)
        if df_clean.empty:
            st.error("No valid data available for Parameters vs Chainage plot after cleaning.")
            return

        # Sort the data by chainage column
        df_clean = df_clean.sort_values(by=chainage_column)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5',
                  '#9B6B6B', '#E9967A', '#4682B4', '#6B8E23']  # Expanded color palette

        available_features = [f for f in selected_features if f in df_clean.columns]
        
        if not available_features:
            st.warning("None of the selected features are available in the dataset.")
            return

        fig = make_subplots(rows=len(available_features), cols=1,
                            shared_xaxes=True,
                            subplot_titles=available_features,
                            vertical_spacing=0.05)  # Reduce spacing between subplots

        for i, feature in enumerate(available_features, start=1):
            try:
                y_data = df_clean[feature]
                feature_name = feature

                fig.add_trace(
                    go.Scatter(
                        x=df_clean[chainage_column],
                        y=y_data,
                        mode='lines',
                        name=feature_name,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ),
                    row=i,
                    col=1
                )

                # Update y-axis titles
                fig.update_yaxes(title_text=feature_name, row=i, col=1)
            except Exception as e:
                st.warning(f"Error plotting feature '{feature}': {e}")

        # Update layout with larger dimensions and better spacing
        fig.update_layout(
            height=min(400 * len(available_features), 800),  # Cap the height at 800px
            width=1200,  # Increased overall width
            title_text=f'Parameters vs Chainage',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100, l=100, r=50, b=50)  # Adjusted margins
        )

        # Update x-axis title only for the bottom subplot
        fig.update_xaxes(title_text='Chainage [mm]', row=len(available_features), col=1)

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating Parameters vs Chainage plot: {e}")

# Updated function to create multi-axis box plots
def create_multi_axis_box_plots(df, selected_features):
    try:
        st.write("Creating Box Plots...")
        if not selected_features:
            st.warning("Please select at least one feature for the box plots.")
            return

        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        colors = ['#0000cd', '#6495ed', '#4b0082', '#ff00ff']  # Corresponding colors

        for i, feature in enumerate(selected_features):
            if i < len(selected_features) // 2:
                fig.add_trace(go.Box(y=df[feature], name=feature, marker_color=colors[i % len(colors)]), secondary_y=False)
            else:
                fig.add_trace(go.Box(y=df[feature], name=feature, marker_color=colors[i % len(colors)]), secondary_y=True)

        fig.update_layout(
            title='Box Plots of Key Parameters',
            height=600,
            width=1000,
            showlegend=True,
            boxmode='group'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating box plots: {e}")

# Updated function to create multi-axis violin plots
def create_multi_axis_violin_plots(df, selected_features):
    try:
        st.write("Creating Violin Plots...")
        if not selected_features:
            st.warning("Please select at least one feature for the violin plots.")
            return

        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        colors = ['#0000cd', '#6495ed', '#4b0082', '#ff00ff']  # Corresponding colors

        for i, feature in enumerate(selected_features):
            if i < len(selected_features) // 2:
                fig.add_trace(go.Violin(y=df[feature], name=feature, box_visible=True, meanline_visible=True, fillcolor=colors[i % len(colors)]), secondary_y=False)
            else:
                fig.add_trace(go.Violin(y=df[feature], name=feature, box_visible=True, meanline_visible=True, fillcolor=colors[i % len(colors)]), secondary_y=True)

        fig.update_layout(
            title='Violin Plots of Key Parameters',
            height=600,
            width=1000,
            showlegend=True,
            violinmode='group'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating violin plots: {e}")

# Function to set background color
def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: rgb(220, 234, 197);
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to add logo
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
        st.error(f"Failed to add logo: {e}")

# Helper function to suggest column based on keywords
def suggest_column(df, keywords):
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return None

# Safe selectbox to handle default selections
def safe_selectbox(label, options, suggested_option):
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
        set_background_color()
        add_logo()

        st.title("Herrenknecht Hard Rock Data Analysis App")

        st.sidebar.header("Data Upload & Analysis")

        uploaded_file = st.sidebar.file_uploader("Machine Data (CSV/Excel)", type=['csv', 'xlsx'])
        rock_strength_file = st.sidebar.file_uploader("Rock Strength Data (CSV/Excel)", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df = load_data(uploaded_file)

            if df is not None:
                working_pressure_cols, revolution_cols, advance_rate_cols = identify_special_columns(df)

                suggested_working_pressure = suggest_column(df, ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'sr_arbdr_z'])
                suggested_revolution = suggest_column(df, ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'sr_drehz_z'])
                suggested_advance_rate = suggest_column(df, ['advance rate', 'vorschub', 'penetration rate', 'vtgeschw', 'geschw', 'geschw_z'])

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

                distance_columns = get_distance_columns(df)
                if not distance_columns:
                    distance_columns = df.columns.tolist()
                selected_distance = st.sidebar.selectbox("Select distance/chainage column", distance_columns)

                n1 = st.sidebar.number_input("Enter n1 value (revolution 1/min)", min_value=0.0, value=1.0, step=0.1)
                torque_constant = st.sidebar.number_input("Enter torque constant", min_value=0.0, value=1.0, step=0.1)

                df = calculate_derived_features(df, working_pressure_col, revolution_col, n1, torque_constant, selected_distance)

                # Move rename_columns after calculate_derived_features
                df_viz = rename_columns(df.copy(), working_pressure_col, revolution_col, selected_distance, advance_rate_col)

                st.write("Processed DataFrame for Visualization:")
                st.write(df_viz.head())

                all_features = df_viz.columns.tolist()
                
                # Allow user to select features, with derived features pre-selected
                default_features = ['Calculated torque [kNm]', 'Average Speed (mm/min)', 'Penetration Rate [mm/rev]']
                default_selected = [feature for feature in default_features if feature in all_features]
                selected_features = st.sidebar.multiselect(
                    "Select features for analysis",
                    all_features,
                    default=default_selected
                )

                time_column = get_time_column(df_viz)

                options = ['Correlation Heatmap', 'Statistical Summary', 'Parameters vs Chainage', 'Box Plots', 'Violin Plots', 'Thrust Force Plots']
                if time_column:
                    options.extend(['Features vs Time', 'Pressure Distribution'])
                if rock_strength_file:
                    options.append('Rock Strength Comparison')

                selected_option = st.sidebar.radio("Choose visualization", options)

                rock_df = None
                if rock_strength_file:
                    rock_strength_data = read_rock_strength_data(rock_strength_file)
                    if rock_strength_data is not None:
                        rock_df = preprocess_rock_strength_data(rock_strength_data)
                        rock_type = st.sidebar.selectbox("Select Rock Type", rock_df.index)

                st.subheader(f"Visualization: {selected_option}")

                if not selected_features and selected_option not in ['Pressure Distribution', 'Thrust Force Plots']:
                    st.warning("Please select at least one feature for analysis.")
                else:
                    if selected_option == 'Correlation Heatmap':
                        create_correlation_heatmap(df_viz, selected_features)
                    elif selected_option == 'Statistical Summary':
                        create_statistical_summary(df_viz, selected_features)
                    elif selected_option == 'Features vs Time' and time_column:
                        create_features_vs_time(df_viz, selected_features, time_column)
                    elif selected_option == 'Pressure Distribution' and time_column:
                        if working_pressure_col and working_pressure_col != 'None':
                            create_pressure_distribution_polar_plot(df_viz, 'Working pressure [bar]', time_column)
                        else:
                            st.warning("Please select a valid working pressure column.")
                    elif selected_option == 'Parameters vs Chainage':
                        create_parameters_vs_chainage(df_viz, selected_features, 'Chainage [mm]')
                    elif selected_option == 'Box Plots':
                        create_multi_axis_box_plots(df_viz, selected_features)
                    elif selected_option == 'Violin Plots':
                        create_multi_axis_violin_plots(df_viz, selected_features)
                    elif selected_option == 'Rock Strength Comparison':
                        if rock_df is not None and 'rock_type' in locals():
                            create_rock_strength_comparison_chart(df_viz, rock_df, rock_type, selected_features)
                        else:
                            st.warning("Please upload rock strength data and select a rock type to view the comparison.")
                    elif selected_option == 'Thrust Force Plots':
                        if advance_rate_col != 'None' and advance_rate_col in df_viz.columns:
                            create_thrust_force_plots(df_viz, advance_rate_col)
                        else:
                            st.warning("Please select a valid advance rate column for Thrust Force Plots.")

                if st.sidebar.button("Download Processed Data"):
                    csv = df_viz.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed CSV File</a>'
                    st.sidebar.markdown(href, unsafe_allow_html=True)

            else:
                st.error("Error loading the data. Please check your file format.")
    except Exception as e:
        st.error(f"An unexpected error occurred in the main function: {str(e)}")

    st.markdown("---")
    st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
    st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

if __name__ == "__main__":
    main()
