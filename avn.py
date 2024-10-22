# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:44:26 2024

@author: KilicK
"""

import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="Herrenknecht Hard Rock Data Analysis App",
    page_icon="https://github.com/kilickursat/AVNanalyzer/blob/main/Herrenknecht_logo.svg-1024x695.png?raw=true"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.interpolate import griddata
from scipy import stats
import matplotlib.pyplot as plt
import base64

# Helper function to clean numeric columns
def clean_numeric_column(df, column_name):
    df[column_name] = df[column_name].replace(r'[^0-9.-]+', '', regex=True)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df[column_name] = df[column_name].fillna(df[column_name].median())
    return df

# Enhanced Function to read CSV or Excel file with validation
def load_data(file):
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
# Function to find the time column based on keywords
def find_time_column(df):
    time_keywords = ['relativzeit', 'zeit', 'Relative time', 'Relative Time', 'time', 'Relativzeit', 'Relativ zeit', 'utc']
    for keyword in time_keywords:
        matches = [col for col in df.columns if keyword.lower() in col.lower()]
        if matches:
            return matches[0]
    return None
    
# Function to aggregate data to per-second level
def aggregate_data(df, sampling_rate):
    if sampling_rate == 'millisecond' or sampling_rate == '10 millisecond':
        df['Relative time'] = pd.to_datetime(df['Relative time'], unit='ms')
        df.set_index('Relative time', inplace=True)
        df = df.resample('1S').mean().reset_index()
    return df
    
def read_rock_strength_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"An error occurred while reading rock strength data: {e}")
        return None

# Function to preprocess the rock strength data
def preprocess_rock_strength_data(df):
    df['Rock Type'] = df['Probenbezeichnung'].str.split().str[0]
    pivoted = df.pivot_table(values='Value', index='Rock Type', columns='Test', aggfunc='mean')
    pivoted.rename(columns={'UCS': 'UCS (MPa)', 'BTS': 'BTS (MPa)', 'PLT': 'PLT (MPa)'}, inplace=True)
    return pivoted

# Function to create comparison chart for machine parameters vs rock strength
def create_rock_strength_comparison_chart(machine_df, rock_df, rock_type):
    rock_df = rock_df[rock_df.index == rock_type]
    if rock_df.empty:
        st.error(f"No data available for {rock_type} rock type.")
        return

    avg_advance_rate = machine_df['Advance rate (mm/min)'].mean()
    avg_penetration_rate = machine_df['Penetration_Rate'].mean()
    avg_torque = machine_df['Calculated torque [kNm]'].mean()
    avg_pressure = machine_df['Working pressure [bar]'].mean()

    parameters = ['Advance Rate\n(mm/min)', 'Penetration Rate\n(mm/rev)', 'Torque\n(kNm)', 'Working Pressure\n(bar)']
    machine_values = [avg_advance_rate, avg_penetration_rate, avg_torque, avg_pressure]
    ucs_values = [rock_df['UCS (MPa)'].iloc[0]] * 4
    bts_values = [rock_df['BTS (MPa)'].iloc[0]] * 4
    plt_values = [rock_df['PLT (MPa)'].iloc[0]] * 4

    fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=600)
    fig.suptitle(f"Machine Parameters vs {rock_type} Rock Strength", fontsize=20, fontweight='bold')

    colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5']

    for i, (param, ax) in enumerate(zip(parameters, axs.flat)):
        x = np.arange(4)
        width = 0.2

        bars = ax.bar(x, [machine_values[i], ucs_values[i], bts_values[i], plt_values[i]], width, color=colors, edgecolor='black')

        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{rock_type} - {param}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)

        machine_param_name = param.split('\n')[0]
        ax.set_xticklabels([machine_param_name, 'UCS', 'BTS', 'PLT'], fontsize=12, fontweight='bold')

        ax.tick_params(axis='y', labelsize=10)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    st.pyplot(fig)

# Function to visualize correlation heatmap with dynamic input
def create_correlation_heatmap(df):
    features = st.multiselect("Select features for correlation heatmap", df.columns, default=[
        'Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Calculated torque [kNm]', 'Penetration_Rate', 'Working pressure [bar]'
    ])
    if len(features) < 2:
        st.warning("Please select at least two features.")
        return
    
    corr_matrix = df[features].corr()
    fig = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    fig.figure.set_size_inches(12, 10)
    fig.set_title('Correlation Heatmap of Selected Parameters')
    st.pyplot(fig.figure)

def create_statistical_summary(df, round_to=2):
    features = st.multiselect("Select features for statistical summary", df.columns, default=[
        'Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]'
    ])
    if not features:
        st.warning("Please select at least one feature.")
        return

    summary_dict = {}
    for feature in features:
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
    
    # Style the table
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
    
# Function to create Features vs Time plot with Plotly subplots
def create_features_vs_time(df, time_col):
    features = st.multiselect("Select features for Time Series plot", df.columns, default=[
        'Revolution [rpm]', 'Thrust force [kN]', 'Calculated torque [kNm]', 'Penetration_Rate', 'Working pressure [bar]'
    ])
    if not features:
        st.warning("Please select at least one feature.")
        return
    
    colors = {
        'Penetration_Rate': '#98fb98', # palegreen
        'Calculated torque [kNm]': '#4b0082', # indigo
        'Working pressure [bar]': '#ff00ff', # magenta
        'Revolution [rpm]': '#0000cd', # mediumblue
        'Thrust force [kN]': '#6495ed' # cornflowerblue
    }

    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, subplot_titles=features)
    for i, feature in enumerate(features, start=1):
        fig.add_trace(go.Scatter(x=df[time_col], y=df[feature], mode='lines', name=feature, line=dict(color=colors.get(feature, '#000000'))), row=i, col=1)

    fig.update_layout(height=300 * len(features), width=1000, title_text='Features vs Time')
    st.plotly_chart(fig)


# Function to create Pressure Distribution Over Time Polar Plot with Plotly (using scatter plots)
def create_pressure_distribution_polar_plot(df):
    pressure_column = 'Working pressure [bar]'
    time_normalized = np.linspace(0, 360, len(df))  # Normalizing time to 360 degrees
    df[pressure_column] = pd.to_numeric(df[pressure_column], errors='coerce')

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df[pressure_column],
        theta=time_normalized,
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
                tickmode='linear',
                tick0=0,
                dtick=45,
                showline=False,
                showgrid=True,
                gridcolor='lightgrey',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°', '360°'],
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

# Function to create Parameters vs Chainage plot with Plotly subplots
def create_parameters_vs_chainage(df):
    features = st.multiselect("Select features for Chainage plot", df.columns, default=[
        'Revolution [rpm]', 'Thrust force [kN]', 'Calculated torque [kNm]', 'Penetration_Rate'
    ])
    if not features:
        st.warning("Please select at least one feature.")
        return

    df = df.sort_values(by='Chainage')  # Sort the data by Chainage

    colors = {
        'Penetration_Rate': '#98fb98', # palegreen
        'Calculated torque [kNm]': '#4b0082', # indigo
        'Revolution [rpm]': '#0000cd', # mediumblue
        'Thrust force [kN]': '#6495ed' # cornflowerblue
    }
    
    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, subplot_titles=features)
    for i, feature in enumerate(features, start=1):
        fig.add_trace(go.Scatter(x=df['Chainage'], y=df[feature], mode='lines', name=feature, line=dict(color=colors.get(feature, '#000000'))), row=i, col=1)

    fig.update_layout(height=300 * len(features), width=1000, title_text='Parameters vs Chainage')
    st.plotly_chart(fig)

# Function to create multi-axis box plots with additional features
def create_multi_axis_box_plots(df):
    features = st.multiselect("Select features for box plots", df.columns, default=[
        'Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]'
    ])
    if not features:
        st.warning("Please select at least one feature.")
        return
    
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    colors = ['#0000cd', '#6495ed', '#4b0082', '#ff00ff']  # Corresponding colors

    for i, feature in enumerate(features):
        if i < len(features) // 2:
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

# Function to create multi-axis violin plots with added customization
def create_multi_axis_violin_plots(df):
    features = st.multiselect("Select features for violin plots", df.columns, default=[
        'Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]'
    ])
    if not features:
        st.warning("Please select at least one feature.")
        return
    
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    colors = ['#0000cd', '#6495ed', '#4b0082', '#ff00ff']  # Corresponding colors

    for i, feature in enumerate(features):
        if i < len(features) // 2:
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
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-image: url(https://github.com/kilickursat/AVNanalyzer/blob/main/Herrenknecht_logo.svg-1024x695.png?raw=true);
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
# Streamlit app
def main():
    st.title("Herrenknecht Hard Rock Data Analysis App")

    # Sidebar for file upload and visualization selection
    st.sidebar.header("Data Upload & Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Machine Data (CSV/Excel)", type=['csv', 'xlsx'])
    rock_strength_file = st.sidebar.file_uploader("Rock Strength Data (Excel)", type=['xlsx'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            # Find time column based on keywords or let user select
            time_col = find_time_column(df)
            if not time_col:
                time_col = st.sidebar.selectbox("Select Relative Time Column", df.columns)
            else:
                st.sidebar.write(f"Auto-detected Time Column: {time_col}")

            # Allow user to select sampling rate for aggregation
            sampling_rate = st.sidebar.selectbox("Select Raw Data Sampling Rate", ['millisecond', '10 millisecond', 'second', 'minute'])
            
            # Aggregate data if required
            if sampling_rate in ['millisecond', '10 millisecond']:
                df = aggregate_data(df, time_col, sampling_rate)

            # Allow user to select column mappings
            st.sidebar.subheader("Column Selection")
            working_pressure_col = st.sidebar.selectbox("Select Working Pressure Column", df.columns)
            revolution_col = st.sidebar.selectbox("Select Revolution Column", df.columns)
            thrust_force_col = st.sidebar.selectbox("Select Thrust Force Column", df.columns)
            advance_rate_col = st.sidebar.selectbox("Select Advance Rate Column", df.columns)
            chainage_col = st.sidebar.selectbox("Select Chainage Column", df.columns)

            # Rename columns based on user selection
            df = df.rename(columns={
                time_col: 'Relative time',
                working_pressure_col: 'Working pressure [bar]',
                revolution_col: 'Revolution [rpm]',
                thrust_force_col: 'Thrust force [kN]',
                advance_rate_col: 'Advance rate (mm/min)',
                chainage_col: 'Chainage'
            })

            numeric_columns = ['Working pressure [bar]', 'Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Advance rate (mm/min)']
            for col in numeric_columns:
                df = clean_numeric_column(df, col)

            # Calculate additional parameters
            df['Calculated torque [kNm]'] = df['Working pressure [bar]'] * 0.1 * 3.14159 / 20
            df['Penetration_Rate'] = df['Advance rate (mm/min)'] / df['Revolution [rpm]']

            # Visualization selection
            options = st.sidebar.radio("Choose visualization", [
                'Features vs Time', 'Correlation Heatmap', 'Statistical Summary'
            ])

            # Visualization based on user selection
            if options == 'Features vs Time':
                create_features_vs_time(df, 'Relative time')
            elif options == 'Correlation Heatmap':
                features = st.multiselect("Select features for correlation heatmap", df.columns, default=[
                    'Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Calculated torque [kNm]', 'Penetration_Rate', 'Working pressure [bar]'
                ])
                if len(features) < 2:
                    st.warning("Please select at least two features.")
                else:
                    corr_matrix = df[features].corr()
                    fig = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                    fig.figure.set_size_inches(12, 10)
                    fig.set_title('Correlation Heatmap of Selected Parameters')
                    st.pyplot(fig.figure)
            elif options == 'Statistical Summary':
                features = st.multiselect("Select features for statistical summary", df.columns, default=[
                    'Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]'
                ])
                if not features:
                    st.warning("Please select at least one feature.")
                else:
                    summary_dict = {}
                    for feature in features:
                        summary_dict[feature] = {
                            'count': int(df[feature].count()),
                            'mean': round(df[feature].mean(), 2),
                            'median': round(df[feature].median(), 2),
                            'std': round(df[feature].std(), 2),
                            'min': round(df[feature].min(), 2),
                            '25%': round(df[feature].quantile(0.25), 2),
                            '50%': round(df[feature].quantile(0.50), 2),
                            '75%': round(df[feature].quantile(0.75), 2),
                            'max': round(df[feature].max(), 2),
                            'skewness': round(df[feature].skew(), 2),
                            'kurtosis': round(df[feature].kurtosis(), 2)
                        }

                    summary = pd.DataFrame(summary_dict).transpose()
                    st.dataframe(summary)

    else:
        st.info("Please upload a machine data file to begin analysis.")

    # Add footer
    st.markdown("---")
    st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
    st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

if __name__ == "__main__":
    main()

