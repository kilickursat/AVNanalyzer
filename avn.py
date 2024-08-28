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

# Function to create statistical summary with additional stats
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
    st.dataframe(summary)

# Function to create Features vs Time plot with Plotly subplots
def create_features_vs_time(df):
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
        fig.add_trace(go.Scatter(x=df['Relative time'], y=df[feature], mode='lines', name=feature, line=dict(color=colors.get(feature, '#000000'))), row=i, col=1)

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
    set_background_color()
    add_logo()

    st.title("Herrenknecht Hard Rock Data Analysis App")

    # Sidebar for file upload and visualization selection
    st.sidebar.header("Data Upload & Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Machine Data (CSV/Excel)", type=['csv', 'xlsx'])
    rock_strength_file = st.sidebar.file_uploader("Rock Strength Data (Excel)", type=['xlsx'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            # Preprocess and clean the data
            if uploaded_file.name.endswith('.csv'):
                # Assuming specific column names as per the provided code
                df = df.rename(columns={
                    df.columns[13]: 'Arbeitsdruck',
                    df.columns[7]: 'Drehzahl',
                    df.columns[30]: 'Advance rate (mm/min)',
                    df.columns[17]: 'Thrust force [kN]',
                    df.columns[27]: 'Chainage',
                    df.columns[2]: 'Relative time',
                    df.columns[28]: 'Weg VTP [mm]',
                    'AzV.V13_SR_Pos_Grad | DB    60.DBD   236': 'SR Position [Grad]',
                    'AzV.V13_SR_ArbDr_Z | DB    60.DBD    26': 'Working pressure [bar]',
                    'AzV.V13_SR_Drehz_nach_Abgl_Z | DB    60.DBD    30': 'Revolution [rpm]'
                })

                numeric_columns = ['Working pressure [bar]', 'Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Advance rate (mm/min)', 'Weg VTP [mm]']
                for col in numeric_columns:
                    df = clean_numeric_column(df, col)

            # Calculate additional parameters
            df['Calculated torque [kNm]'] = df['Working pressure [bar]'] * 0.1 * 3.14159 / 20
            df['Penetration_Rate'] = df['Advance rate (mm/min)'] / df['Revolution [rpm]']

            # Visualization selection
            options = st.sidebar.radio("Choose visualization", [
                'Correlation Heatmap', 'Statistical Summary', 
                'Features vs Time', 'Pressure Distribution',
                'Parameters vs Chainage', 'Box Plots', 
                'Violin Plots', 'Rock Strength Comparison'
            ])

            # Rock strength data processing
            rock_df = None
            if rock_strength_file:
                rock_strength_data = read_rock_strength_data(rock_strength_file)
                if rock_strength_data is not None:
                    rock_df = preprocess_rock_strength_data(rock_strength_data)
                    rock_type = st.sidebar.selectbox("Select Rock Type", rock_df.index)

            # Visualization based on user selection
            if options == 'Correlation Heatmap':
                create_correlation_heatmap(df)
            elif options == 'Statistical Summary':
                create_statistical_summary(df)
            elif options == 'Features vs Time':
                create_features_vs_time(df)
            elif options == 'Pressure Distribution':
                create_pressure_distribution_polar_plot(df)
            elif options == 'Parameters vs Chainage':
                create_parameters_vs_chainage(df)
            elif options == 'Box Plots':
                create_multi_axis_box_plots(df)
            elif options == 'Violin Plots':
                create_multi_axis_violin_plots(df)
            elif options == 'Rock Strength Comparison':
                if rock_df is not None and 'rock_type' in locals():
                    create_rock_strength_comparison_chart(df, rock_df, rock_type)
                else:
                    st.warning("Please upload rock strength data and select a rock type to view the comparison.")

            # Add an option to download the processed data
            if st.sidebar.button("Download Processed Data"):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed CSV File</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)

    else:
        st.info("Please upload a machine data file to begin analysis.")

    # Add footer
    st.markdown("---")
    st.markdown("© 2023 Herrenknecht AG. All rights reserved.")

if __name__ == "__main__":
    main()
