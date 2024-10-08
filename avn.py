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
import io
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data = None

def init_page_config():
    """Initialize Streamlit page configuration"""
    try:
        st.set_page_config(
            page_title="Herrenknecht Hard Rock Data Analysis App",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        logger.error(f"Error setting page configuration: {e}")
        st.error(f"Error setting page configuration: {e}")

def set_background_color():
    """Set the background color of the app"""
    try:
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
    except Exception as e:
        logger.error(f"Error setting background color: {e}")

def add_logo():
    """Add the Herrenknecht logo to the sidebar"""
    try:
        st.sidebar.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                background-image: url(https://github.com/kilickursat/AVNanalyzer/blob/main/Herrenknecht_logo.svg-1024x695.png?raw=true);
                background-repeat: no-repeat;
                background-size: 120px;
                background-position: 10px 10px;
                padding-top: 120px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.error(f"Error adding logo: {e}")

@st.cache_data
def clean_numeric_column(df, column_name):
    """Clean numeric columns by removing non-numeric characters and handling missing values"""
    try:
        df[column_name] = df[column_name].replace(r'[^0-9.-]+', '', regex=True)
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        df[column_name] = df[column_name].fillna(df[column_name].median())
        return df
    except Exception as e:
        logger.error(f"Error cleaning numeric column {column_name}: {e}")
        return df

@st.cache_data
def load_data(file):
    """Load and preprocess the data file"""
    try:
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'])
            except:
                df = pd.read_csv(file, sep=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            return None

        if df.empty:
            st.error("The uploaded file is empty or not formatted correctly.")
            return None

        # Clean numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            df = clean_numeric_column(df, column)

        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def read_rock_strength_data(file):
    """Read and process rock strength data"""
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        logger.error(f"Error reading rock strength data: {e}")
        st.error(f"Error reading rock strength data: {e}")
        return None

def calculate_derived_features(df, working_pressure_col, advance_rate_col, revolution_col, n1, torque_constant):
    """Calculate derived features from machine parameters"""
    try:
        if working_pressure_col:
            df['Calculated_Torque'] = df[working_pressure_col] * torque_constant
            
        if advance_rate_col and revolution_col:
            # Convert advance rate from mm/min to mm/rev
            df['Penetration_Rate'] = df[advance_rate_col] / (df[revolution_col] * n1)
            
        return df
    except Exception as e:
        logger.error(f"Error calculating derived features: {e}")
        st.error(f"Error calculating derived features: {e}")
        return df

def debug_info(df):
    """Display debugging information"""
    try:
        with st.expander("Debug Information"):
            st.write("DataFrame Info:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.write("DataFrame Head:")
            st.write(df.head())
            
            st.write("DataFrame Columns:")
            st.write(df.columns.tolist())
    except Exception as e:
        logger.error(f"Error displaying debug info: {e}")
        st.error(f"Error displaying debug info: {e}")

def identify_special_columns(df):
    """Identify special columns in the dataset"""
    try:
        working_pressure_keywords = ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr', 'SR_Arbdr']
        revolution_keywords = ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz']
        advance_rate_keywords = ['advance rate', 'vortrieb', 'vorschub', 'penetration rate', 'VTgeschw_Z', 'geschw', 'geschw_Z']

        working_pressure_cols = [col for col in df.columns if any(kw in col.lower() for kw in working_pressure_keywords)]
        revolution_cols = [col for col in df.columns if any(kw in col.lower() for kw in revolution_keywords)]
        advance_rate_cols = [col for col in df.columns if any(kw in col.lower() for kw in advance_rate_keywords)]

        return working_pressure_cols, revolution_cols, advance_rate_cols
    except Exception as e:
        logger.error(f"Error identifying special columns: {e}")
        return [], [], []

# Visualization Functions
def create_correlation_heatmap(df, selected_features):
    """Create and display correlation heatmap"""
    try:
        if len(selected_features) < 2:
            st.warning("Please select at least two features for the correlation heatmap.")
            return
        
        plt.figure(figsize=(12, 10))
        corr_matrix = df[selected_features].corr()
        fig = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        fig.set_title('Correlation Heatmap of Selected Parameters')
        st.pyplot(fig.figure)
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        st.error(f"Error creating correlation heatmap: {e}")

def create_statistical_summary(df, selected_features, round_to=2):
    """Create and display statistical summary"""
    try:
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
                'max': round(df[feature].max(), round_to),
                'skewness': round(df[feature].skew(), round_to),
                'kurtosis': round(df[feature].kurtosis(), round_to)
            }

        summary = pd.DataFrame(summary_dict).transpose()
        st.dataframe(summary)
    except Exception as e:
        logger.error(f"Error creating statistical summary: {e}")
        st.error(f"Error creating statistical summary: {e}")

def create_features_vs_time(df, selected_features, time_column):
    """Create time series plots for selected features"""
    try:
        if not selected_features:
            st.warning("Please select at least one feature for the time series plot.")
            return
        
        fig = make_subplots(rows=len(selected_features), cols=1, 
                           subplot_titles=selected_features,
                           vertical_spacing=0.05)
        
        for i, feature in enumerate(selected_features, start=1):
            fig.add_trace(
                go.Scatter(x=df[time_column], y=df[feature], 
                          mode='lines', name=feature),
                row=i, col=1
            )
            fig.update_yaxes(title_text=feature, row=i, col=1)
        
        fig.update_layout(height=300*len(selected_features), 
                         showlegend=True,
                         title_text='Features vs Time')
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error creating time series plots: {e}")
        st.error(f"Error creating time series plots: {e}")

def create_pressure_distribution_polar_plot(df, pressure_column, time_column):
    """Create polar plot for pressure distribution"""
    try:
        df[pressure_column] = pd.to_numeric(df[pressure_column], errors='coerce')
        df['normalized_time'] = (df[time_column] - df[time_column].min()) / (df[time_column].max() - df[time_column].min()) * 360

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=df[pressure_column],
            theta=df['normalized_time'],
            mode='markers',
            name='Pressure'
        ))

        fig.update_layout(
            title='Pressure Distribution Over Time (Polar Plot)',
            polar=dict(radialaxis=dict(showline=False, range=[0, df[pressure_column].max() * 1.1])),
            showlegend=False
        )

        st.plotly_chart(fig)
    except Exception as e:
        logger.error(f"Error creating pressure distribution plot: {e}")
        st.error(f"Error creating pressure distribution plot: {e}")

def create_rock_strength_comparison_chart(machine_df, rock_df, rock_type, selected_features):
    """Create comparison chart for machine parameters vs rock strength"""
    try:
        if rock_df[rock_df.index == rock_type].empty:
            st.error(f"No data available for {rock_type} rock type.")
            return

        avg_values = machine_df[selected_features].mean()
        rock_strength = rock_df.loc[rock_type]

        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"Machine Parameters vs {rock_type} Rock Strength")

        for i, (feature, ax) in enumerate(zip(selected_features, axs.flat)):
            values = [avg_values[feature], rock_strength['UCS (MPa)'], 
                     rock_strength['BTS (MPa)'], rock_strength['PLT (MPa)']]
            ax.bar(['Machine', 'UCS', 'BTS', 'PLT'], values)
            ax.set_title(feature)

        st.pyplot(fig)
    except Exception as e:
        logger.error(f"Error creating rock strength comparison: {e}")
        st.error(f"Error creating rock strength comparison: {e}")

def main():
    """Main application function"""
    try:
        init_page_config()
        set_background_color()
        add_logo()

        st.title("Herrenknecht Hard Rock Data Analysis App")

        # Sidebar setup
        st.sidebar.header("Data Upload & Analysis")
        uploaded_file = st.sidebar.file_uploader("Machine Data (CSV/Excel)", type=['csv', 'xlsx'])
        rock_strength_file = st.sidebar.file_uploader("Rock Strength Data (Excel)", type=['xlsx'])

        if uploaded_file is not None:
            if not st.session_state.data_loaded:
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.data = df
                    st.session_state.data_loaded = True
            
            if st.session_state.data_loaded:
                df = st.session_state.data
                
                # Debug information
                debug_info(df)
                
                # Column selection and feature calculation
                working_pressure_cols, revolution_cols, advance_rate_cols = identify_special_columns(df)
                
                # Parameter selection
                selected_features = st.sidebar.multiselect("Select features for analysis", df.columns)
                
                # Visualization selection
                viz_options = ['Correlation Heatmap', 'Statistical Summary', 
                             'Features vs Time', 'Pressure Distribution',
                             'Rock Strength Comparison']
                
                selected_viz = st.sidebar.selectbox("Choose visualization", viz_options)
                
                # Display selected visualization
                if selected_viz == 'Correlation Heatmap':
                    create_correlation_heatmap(df, selected_features)
                elif selected_viz == 'Statistical Summary':
                    create_statistical_summary(df, selected_features)
                elif selected_viz == 'Features vs Time':
                    time_col = st.sidebar.selectbox("Select time column", df.columns)
                    if time_col:
                        create_features_vs_time(df, selected_features, time_col)
                elif selected_viz == 'Pressure Distribution':
                    pressure_col = st.sidebar.selectbox("Select pressure column", working_pressure_cols)
                    time_col = st.sidebar.selectbox("Select time column", df.columns)
                    if pressure_col and time_col:
                        create_pressure_distribution_polar_plot(df, pressure_col, time_col)
                elif selected_viz == 'Rock Strength Comparison':
                    if rock_strength_file:
                        rock_df = read_rock_strength_data(rock_strength_file)
                        if rock_df is not None:
                            rock_type = st.sidebar.selectbox("Select Rock Type", rock_df.index)
                            create_rock_strength_comparison_chart(df, rock_df, rock_type, selected_features)
                
                # Download processed data
                if st.sidebar.button("Download Processed Data"):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed CSV File</a>'
                    st.sidebar.markdown(href, unsafe_allow_html=True)
        
        else:
            st.info("Please upload a machine data file to begin analysis.")
        
        # Footer
        st.markdown("---")
        st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
        st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")
        
    except Exception as e:
        logger.error(f"Main application error: {e}")
        st.error(f"An error occurred in the main application: {e}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
