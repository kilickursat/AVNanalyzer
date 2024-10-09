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
)

# Helper function to clean numeric columns
def clean_numeric_column(df, column_name):
    df[column_name] = df[column_name].replace(r'[^0-9.-]+', '', regex=True)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df[column_name] = df[column_name].fillna(df[column_name].median())
    return df

# Advanced rate calculation function
def calculate_advance_rate_and_stats(df, distance_column, time_column):
    try:
        if not all(col in df.columns for col in [distance_column, time_column]):
            raise ValueError(f"Required columns not found in DataFrame")
            
        if len(df) > 1:
            weg = round(df[distance_column].max() - df[distance_column].min(), 2)
            zeit = round(df[time_column].max() - df[time_column].min(), 2)
        else:
            weg = round(df[distance_column].iloc[0], 2)
            zeit = round(df[time_column].iloc[0], 2)
            
        zeit = zeit * (0.000001 / 60)
        
        average_speed = round(weg / zeit, 2) if zeit != 0 else 0
        
        result = {
            "Total Distance (mm)": weg,
            "Total Time (min)": zeit,
            "Average Speed (mm/min)": average_speed
        }
        
        return result, average_speed
        
    except Exception as e:
        st.error(f"Error in advance rate calculation: {e}")
        return None, 0

# Penetration rate calculation function
def calculate_penetration_rate(row):
    try:
        speed = row['Average Speed (mm/min)']
        revolution = row['Revolution [rpm]']
        
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
        if working_pressure_col is not None and revolution_col is not None:
            df[working_pressure_col] = pd.to_numeric(df[working_pressure_col], errors='coerce')
            df[revolution_col] = pd.to_numeric(df[revolution_col], errors='coerce')

            def calculate_torque_wrapper(row):
                working_pressure = row[working_pressure_col]
                current_speed = row[revolution_col]

                if pd.isna(working_pressure) or pd.isna(current_speed):
                    return np.nan

                if current_speed < n1:
                    torque = working_pressure * torque_constant
                else:
                    torque = (n1 / current_speed) * torque_constant * working_pressure

                return round(torque, 2)

            df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)
        
        # Calculate advance rate and average speed
        distance_column = selected_distance
        time_column = get_time_column(df)
        
        if distance_column in df.columns and time_column:
            result, average_speed = calculate_advance_rate_and_stats(df, distance_column, time_column)
            df['Average Speed (mm/min)'] = average_speed
            
            if revolution_col is not None:
                df['Penetration Rate [mm/rev]'] = df.apply(lambda row: calculate_penetration_rate(row, revolution_col), axis=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating derived features: {str(e)}")
        return df


# Helper functions for column identification
def identify_special_columns(df):
    working_pressure_keywords = ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'SR_Arbdr']
    revolution_keywords = ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz']
    advance_rate_keywords = ['advance rate', 'vortrieb', 'vorschub', 'penetration rate', 'VTgeschw_Z', 'geschw', 'geschw_Z']

    working_pressure_cols = [col for col in df.columns if any(kw in col.lower() for kw in working_pressure_keywords)]
    revolution_cols = [col for col in df.columns if any(kw in col.lower() for kw in revolution_keywords)]
    advance_rate_cols = [col for col in df.columns if any(kw in col.lower() for kw in advance_rate_keywords)]

    return working_pressure_cols, revolution_cols, advance_rate_cols

def get_distance_columns(df):
    distance_keywords = ['distance', 'length', 'travel', 'chainage', 'Tunnellänge Neu', 'Tunnellänge', 'Weg_mm_Z', 'VTP_Weg']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in distance_keywords)]

def get_time_column(df):
    time_keywords = ['relativzeit', 'relative time', 'time', 'datum', 'date', 'zeit', 'timestamp', 'Relative Time', 'Relativzeit']
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
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format")
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
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error reading rock strength data: {e}")
        return None

# Function to preprocess the rock strength data
def preprocess_rock_strength_data(df):
    try:
        df['Rock Type'] = df['Probenbezeichnung'].str.split().str[0]
        pivoted = df.pivot_table(values='Value', index='Rock Type', columns='Test', aggfunc='mean')
        pivoted.rename(columns={'UCS': 'UCS (MPa)', 'BTS': 'BTS (MPa)', 'PLT': 'PLT (MPa)'}, inplace=True)
        return pivoted
    except Exception as e:
        st.error(f"Error preprocessing rock strength data: {e}")
        return None


# Updated function to create comparison chart for machine parameters vs rock strength
def create_rock_strength_comparison_chart(machine_df, rock_df, rock_type, selected_features):
    rock_df = rock_df[rock_df.index == rock_type]
    if rock_df.empty:
        st.error(f"No data available for {rock_type} rock type.")
        return

    avg_values = machine_df[selected_features].mean()

    parameters = []
    machine_values = []
    for feature in selected_features:
        if any(keyword in feature.lower() for keyword in ['advance rate', 'vortrieb', 'vorschub', 'VTgeschw', 'geschw']):
            parameters.append('Advance rate [mm/min]')
        elif any(keyword in feature.lower() for keyword in ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz']):
            parameters.append('Revolution [rpm]')
        elif any(keyword in feature.lower() for keyword in ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'SR_Arbdr']):
            parameters.append('Working pressure [bar]')
        else:
            parameters.append(feature)
        machine_values.append(avg_values[feature])

    ucs_values = [rock_df['UCS (MPa)'].iloc[0]] * len(selected_features)
    bts_values = [rock_df['BTS (MPa)'].iloc[0]] * len(selected_features)
    plt_values = [rock_df['PLT (MPa)'].iloc[0]] * len(selected_features)

    fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=100)  # Reduced dpi from 600 to 100
    fig.suptitle(f"Machine Parameters vs {rock_type} Rock Strength", fontsize=20, fontweight='bold')

    colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5']

    for i, (param, ax) in enumerate(zip(parameters, axs.flat)):
        x = np.arange(4)
        width = 0.2

        bars = ax.bar(x, [machine_values[i], ucs_values[i], bts_values[i], plt_values[i]], width, color=colors, edgecolor='black')

        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{rock_type} - {param}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)

        ax.set_xticklabels([param, 'UCS', 'BTS', 'PLT'], fontsize=12, fontweight='bold')

        ax.tick_params(axis='y', labelsize=10)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    st.pyplot(fig)

# Updated function to visualize correlation heatmap with dynamic input
def create_correlation_heatmap(df, selected_features):
    if len(selected_features) < 2:
        st.warning("Please select at least two features for the correlation heatmap.")
        return

    corr_matrix = df[selected_features].corr()
    fig = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    fig.figure.set_size_inches(12, 10)
    fig.set_title('Correlation Heatmap of Selected Parameters')
    st.pyplot(fig.figure)

# Updated function to create statistical summary
def create_statistical_summary(df, selected_features, round_to=2):
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

def create_features_vs_time(df, selected_features, time_column):
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


# Updated function to create Pressure Distribution Over Time Polar Plot with Plotly
def create_pressure_distribution_polar_plot(df, pressure_column, time_column):
    try:
        df[pressure_column] = pd.to_numeric(df[pressure_column], errors='coerce')

        # Normalize time to 360 degrees
        df['normalized_time'] = (df[time_column] - df[time_column].min()) / (df[time_column].max() - df[time_column].min()) * 360

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=df[pressure_column],
            theta=df['normalized_time'],
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Pressure'
        ))

        max_pressure = df[pressure_column].max()
        if pd.isna(max_pressure):
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

def create_parameters_vs_chainage(df, selected_features, chainage_column):
    if not selected_features:
        st.warning("Please select at least one feature for the chainage plot.")
        return

    # Ensure the chainage column exists
    if chainage_column not in df.columns:
        st.error(f"Chainage column '{chainage_column}' not found in the dataset.")
        return

    # Sort the data by chainage column
    df = df.sort_values(by=chainage_column)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5',
              '#9B6B6B', '#E9967A', '#4682B4', '#6B8E23']  # Expanded color palette

    fig = make_subplots(rows=len(selected_features), cols=1,
                        shared_xaxes=True,
                        subplot_titles=selected_features,
                        vertical_spacing=0.05)  # Reduce spacing between subplots

    for i, feature in enumerate(selected_features, start=1):
        try:
            y_data = df[feature]
            feature_name = feature

            # Replace sensor names with standardized names
            if any(keyword in feature.lower() for keyword in ['advance rate', 'vortrieb', 'vorschub', 'VTgeschw', 'geschw']):
                feature_name = 'Advance rate [mm/min]'
            elif any(keyword in feature.lower() for keyword in ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz']):
                feature_name = 'Revolution [rpm]'
            elif any(keyword in feature.lower() for keyword in ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'SR_Arbdr']):
                feature_name = 'Working pressure [bar]'

            fig.add_trace(
                go.Scatter(
                    x=df[chainage_column],
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
            st.error(f"Error plotting feature '{feature}': {e}")

    # Update layout with larger dimensions and better spacing
    fig.update_layout(
        height=400 * len(selected_features),  # Increased height per subplot
        width=1200,  # Increased overall width
        title_text=f'Parameters vs {chainage_column}',
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
    fig.update_xaxes(title_text=chainage_column, row=len(selected_features), col=1)

    st.plotly_chart(fig, use_container_width=True)

# Function to get distance-related columns
def get_distance_columns(df):
    distance_keywords = ['distance', 'length', 'travel', 'chainage', 'Tunnellänge Neu', 'Tunnellänge','Weg_mm_Z','VTP_Weg']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in distance_keywords)]

# Updated function to create multi-axis box plots with additional features
def create_multi_axis_box_plots(df, selected_features):
    if not selected_features:
        st.warning("Please select at least one feature for the box plots.")
        return

    try:
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

# Updated function to create multi-axis violin plots with added customization
def create_multi_axis_violin_plots(df, selected_features):
    if not selected_features:
        st.warning("Please select at least one feature for the violin plots.")
        return

    try:
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

def identify_special_columns(df):
    working_pressure_keywords = ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'SR_Arbdr']
    revolution_keywords = ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz']
    advance_rate_keywords = ['advance rate', 'vortrieb', 'vorschub', 'penetration rate', 'VTgeschw_Z','geschw','geschw_Z']

    working_pressure_cols = [col for col in df.columns if any(kw in col.lower() for kw in working_pressure_keywords)]
    revolution_cols = [col for col in df.columns if any(kw in col.lower() for kw in revolution_keywords)]
    advance_rate_cols = [col for col in df.columns if any(kw in col.lower() for kw in advance_rate_keywords)]

    return working_pressure_cols, revolution_cols, advance_rate_cols

# Helper function to suggest column based on keywords
def suggest_column(df, keywords):
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return None

def get_time_column(df):
    time_keywords = ['relativzeit', 'relative time', 'time', 'datum', 'date', 'zeit', 'timestamp', 'Relative Time', 'Relativzeit']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            return col
    return None

def create_thrust_force_plots(df, advance_rate_col):
    try:
        thrust_force_col = suggest_column(df, ['thrust force', 'vorschubkraft', 'kraft','kraft_max','gesamtkraft','gesamtkraft_stz','gesamtkraft_vtp'])
        penetration_rate_col = 'Penetration Rate [mm/rev]'

        if thrust_force_col is None:
            st.warning("Thrust force column not found in the dataset.")
            return

        if penetration_rate_col not in df.columns:
            st.warning("Penetration Rate [mm/rev] column not found. Ensure it is calculated correctly.")
            return

        if advance_rate_col not in df.columns:
            st.warning(f"Selected Advance Rate column '{advance_rate_col}' not found.")
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
            name=f'Advance Rate ({advance_rate_col})',
            marker=dict(color='green', size=5)
        ), row=2, col=1)

        fig.update_layout(
            height=800, 
            width=800, 
            title_text="Thrust Force Relationships"
        )
        fig.update_xaxes(title_text="Penetration Rate [mm/rev]", row=1, col=1)
        fig.update_xaxes(title_text=f"Advance Rate ({advance_rate_col})", row=2, col=1)
        fig.update_yaxes(title_text="Thrust Force [kN]", row=1, col=1)
        fig.update_yaxes(title_text="Thrust Force [kN]", row=2, col=1)

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating thrust force plots: {e}")

def safe_selectbox(label, options, suggested_option):
    try:
        if suggested_option and suggested_option in options:
            index = options.index(suggested_option)
        else:
            index = 0  # Default to 'None'
    except ValueError:
        index = 0  # Default to 'None' if suggested_option is not in options
    return st.sidebar.selectbox(label, options, index=index)

# Add these functions after the existing helper functions and before the visualization functions

def calculate_advance_rate_and_stats(df, distance_column, time_column):
    try:
        df[distance_column] = pd.to_numeric(df[distance_column], errors='coerce')
        df[time_column] = pd.to_numeric(df[time_column], errors='coerce')

        if len(df) > 1:
            weg = round(df[distance_column].max() - df[distance_column].min(), 2)
            zeit = round(df[time_column].max() - df[time_column].min(), 2)
        else:
            weg = df[distance_column].iloc[0]
            zeit = df[time_column].iloc[0]

        zeit = zeit * (0.000001 / 60)

        average_speed = round(weg / zeit, 2) if zeit != 0 else 0

        result = {
            "Total Distance (mm)": weg,
            "Total Time (min)": zeit,
            "Average Speed (mm/min)": average_speed
        }

        return result, average_speed
    except Exception as e:
        st.error(f"Error calculating advance rate stats: {str(e)}")
        return None, 0


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
        st.error(f"Error calculating penetration rate: {str(e)}")
        return np.nan

# Main function
def main():
    try:
        set_background_color()
        add_logo()

        st.title("Herrenknecht Hard Rock Data Analysis App")

        # Sidebar for file upload and visualization selection
        st.sidebar.header("Data Upload & Analysis")

        uploaded_file = st.sidebar.file_uploader("Machine Data (CSV/Excel)", type=['csv', 'xlsx'])
        rock_strength_file = st.sidebar.file_uploader("Rock Strength Data (CSV/Excel)", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df = load_data(uploaded_file)

            if df is not None:
                # Identify special columns
                working_pressure_cols, revolution_cols, advance_rate_cols = identify_special_columns(df)

                # Suggest columns based on keywords
                suggested_working_pressure = suggest_column(df, ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr','SR_Arbdr'])
                suggested_revolution = suggest_column(df, ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz'])

                # Let user select working pressure and revolution columns
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

                # Add input fields for n1 and torque_constant
                n1 = st.sidebar.number_input("Enter n1 value (revolution 1/min)", min_value=0.0, value=1.0, step=0.1)
                torque_constant = st.sidebar.number_input("Enter torque constant", min_value=0.0, value=1.0, step=0.1)

                # Get distance-related columns
                distance_columns = get_distance_columns(df)

                # Force distance column selection
                if not distance_columns:
                    distance_columns = df.columns.tolist()  # Use all columns if no distance columns are detected
                selected_distance = st.sidebar.selectbox("Select distance/chainage column", distance_columns)

                # Calculate derived features if possible
                if working_pressure_col != 'None' and revolution_col != 'None':
                    df = calculate_derived_features(df,
                                                    working_pressure_col,
                                                    revolution_col,
                                                    n1,
                                                    torque_constant,
                                                    selected_distance)

                # Let user select features for analysis
                all_features = df.columns.tolist()
                if 'Penetration Rate [mm/rev]' in all_features:
                    all_features.remove('Penetration Rate [mm/rev]')
                if 'Average Speed (mm/min)' in all_features:
                    all_features.remove('Average Speed (mm/min)')
                selected_features = st.sidebar.multiselect("Select features for analysis", all_features)
                selected_features.extend(['Penetration Rate [mm/rev]', 'Average Speed (mm/min)'])

                # Check for time-related columns
                time_column = get_time_column(df)

                # Visualization selection
                options = ['Correlation Heatmap', 'Statistical Summary', 'Parameters vs Chainage', 'Box Plots', 'Violin Plots', 'Thrust Force Plots']

                if time_column:
                    options.extend(['Features vs Time', 'Pressure Distribution'])

                if rock_strength_file:
                    options.append('Rock Strength Comparison')

                selected_option = st.sidebar.radio("Choose visualization", options)

                # Rock strength data processing
                rock_df = None
                if rock_strength_file:
                    rock_strength_data = read_rock_strength_data(rock_strength_file)
                    if rock_strength_data is not None:
                        rock_df = preprocess_rock_strength_data(rock_strength_data)
                        rock_type = st.sidebar.selectbox("Select Rock Type", rock_df.index)

                # Main content area - Visualization based on user selection
                st.subheader(f"Visualization: {selected_option}")

                if not selected_features and selected_option not in ['Pressure Distribution', 'Thrust Force Plots']:
                    st.warning("Please select at least one feature for analysis.")
                else:
                    if selected_option == 'Correlation Heatmap':
                        create_correlation_heatmap(df, selected_features)
                    elif selected_option == 'Statistical Summary':
                        create_statistical_summary(df, selected_features)
                    elif selected_option == 'Features vs Time' and time_column:
                        create_features_vs_time(df, selected_features, time_column)
                    elif selected_option == 'Pressure Distribution' and time_column:
                        if working_pressure_col and working_pressure_col != 'None':
                            create_pressure_distribution_polar_plot(df, working_pressure_col, time_column)
                        else:
                            st.warning("Please select a valid working pressure column.")
                    elif selected_option == 'Parameters vs Chainage':
                        create_parameters_vs_chainage(df, selected_features, selected_distance)
                    elif selected_option == 'Box Plots':
                        create_multi_axis_box_plots(df, selected_features)
                    elif selected_option == 'Violin Plots':
                        create_multi_axis_violin_plots(df, selected_features)
                    elif selected_option == 'Rock Strength Comparison':
                        if rock_df is not None and 'rock_type' in locals():
                            create_rock_strength_comparison_chart(df, rock_df, rock_type, selected_features)
                        else:
                            st.warning("Please upload rock strength data and select a rock type to view the comparison.")
                    elif selected_option == 'Thrust Force Plots':
                        create_thrust_force_plots(df,advance_rate_col)

                # Add download button for processed data
                if st.sidebar.button("Download Processed Data"):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed CSV File</a>'
                    st.sidebar.markdown(href, unsafe_allow_html=True)

            else:
                st.error("Error loading the data. Please check your file format.")
    except Exception as e:
        st.error(f"An unexpected error occurred in the main function: {str(e)}")

    # Add footer
    st.markdown("---")
    st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
    st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")
    
if __name__ == "__main__":
    main()
