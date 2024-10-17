import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.interpolate import griddata
from scipy import stats
import matplotlib.pyplot as plt
import io
import base64

# Set page config at the very beginning
st.set_page_config(
    page_title="Herrenknecht Hard Rock Data Analysis App",
    page_icon="https://raw.githubusercontent.com/kilickursat/torque_plot-updated/main/Herrenknecht_logo.svg-1024x695.png",
    layout="wide"
)

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
                padding-top: 150px;  /* Increased padding for logo */
            }
            .sidebar-content {
                padding-top: 10px;  /* Reduced padding */
            }
            .sidebar-content > * {
                margin-bottom: 0.5rem !important;
            }
            /* Consistent font sizes for headers */
            .sidebar .sidebar-content div[data-testid="stMarkdownContainer"] > h1 {
                font-size: 1.5em;
                margin-top: 0;
            }
            .sidebar .sidebar-content div[data-testid="stMarkdownContainer"] > h2 {
                font-size: 1.2em;
                margin-top: 0;
            }
            /* Consistent sizing for file uploader */
            .sidebar .sidebar-content [data-testid="stFileUploader"] {
                margin-bottom: 0.5rem;
            }
            /* Consistent spacing for radio buttons */
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
            
        # Assuming 'zeit' is in seconds for consistency
        zeit = zeit / 60  # Convert to minutes
        
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
            
            if revolution_col is not None and revolution_col != 'None':
                df['Penetration Rate [mm/rev]'] = df.apply(lambda row: calculate_penetration_rate(row, revolution_col), axis=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating derived features: {str(e)}")
        return df

# Helper functions for column identification
def identify_special_columns(df):
    working_pressure_keywords = ['working pressure', 'arbeitsdruck', 'sr_arbdr', 'SR_Arbdr','pressure', 'druck', 'arbdr']
    revolution_keywords = ['revolution', 'revolution (rpm)','drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz','Revolution', 'Revolution [rpm]','Revolution (rpm)']
    advance_rate_keywords = ['advance rate', 'advance rate [mm/min]', 'Advance rate', 'Advance_rate','Advance Rate','vortrieb', 'vorschub', 'VTgeschw_Z', 'geschw', 'geschw_Z']

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
            try:
                # Attempt to convert to numeric first (for relative time)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    return col
            except:
                continue
    return None

# Enhanced Function to read CSV or Excel file with validation
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True, engine='python')
            df.columns = df.columns.str.strip()  # Trim whitespace from column names
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
            df.columns = df.columns.str.strip()  # Trim whitespace from column names
        else:
            st.error("Unsupported file format")
            return None

        if df.empty:
            st.error("The uploaded file is empty or not formatted correctly.")
            return None

        # Ensure all columns are properly read
        df.columns = [col.strip() for col in df.columns]

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
def create_rock_strength_comparison_chart(df, rock_df, rock_type, selected_features):
    try:
        # Prepare data for plotting
        rock_strength_data = rock_df.loc[rock_type].dropna()
        machine_data = df[selected_features].mean()

        # Combine rock strength and machine data
        combined_data = pd.concat([rock_strength_data, machine_data])

        # Create the bar chart
        fig = go.Figure()

        # Add bars for rock strength parameters
        fig.add_trace(go.Bar(
            x=rock_strength_data.index,
            y=rock_strength_data.values,
            name='Rock Strength',
            marker_color='#FF6B6B'
        ))

        # Add bars for machine parameters
        fig.add_trace(go.Bar(
            x=machine_data.index,
            y=machine_data.values,
            name='Machine Parameters',
            marker_color='#4ECDC4'
        ))

        # Update layout
        fig.update_layout(
            title=f'Rock Strength ({rock_type}) vs Machine Parameters Comparison',
            xaxis_title="Parameters",
            yaxis_title="Values",
            barmode='group',
            height=600,
            width=1000,
            showlegend=True,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1
            )
        )

        # Improve hover information
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Value: %{y:.2f}<br>" +
                         "<extra></extra>"
        )

        return fig
    except Exception as e:
        st.error(f"Error creating rock strength comparison chart: {e}")
        return None

def rename_columns(df, working_pressure_col, revolution_col, distance_col, advance_rate_col):
    column_mapping = {}
    if working_pressure_col and working_pressure_col != 'None':
        column_mapping[working_pressure_col] = 'Working pressure [bar]'
    if revolution_col and revolution_col != 'None':
        column_mapping[revolution_col] = 'Revolution [rpm]'
    if distance_col and distance_col != 'None':
        column_mapping[distance_col] = 'Chainage [mm]'
    if advance_rate_col and advance_rate_col != 'None':
        column_mapping[advance_rate_col] = 'Advance rate [mm/min]'
    return df.rename(columns=column_mapping)

# Updated function to visualize correlation heatmap with dynamic input
def create_correlation_heatmap(df, selected_features):
    if len(selected_features) < 2:
        st.warning("Please select at least two features for the correlation heatmap.")
        return

    # Only use the features that the user has explicitly selected
    available_features = [f for f in selected_features if f in df.columns]
    
    if len(available_features) < 2:
        st.warning("Please select at least two valid features for the correlation heatmap.")
        return

    try:
        corr_matrix = df[available_features].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title('Correlation Heatmap of Selected Parameters')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")

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

# Updated function to create features vs time
def create_features_vs_time(df, selected_features, time_column, sampling_rate):
    if not selected_features:
        st.warning("Please select at least one feature for the time series plot.")
        return

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5',
              '#9B6B6B', '#E9967A', '#4682B4', '#6B8E23']  # Expanded color palette

    fig = make_subplots(rows=len(selected_features), cols=1,
                        shared_xaxes=True,
                        subplot_titles=selected_features,
                        vertical_spacing=0.05)  # Reduced spacing between subplots

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

    # Update layout with adjusted dimensions
    fig.update_layout(
        height=300 * len(selected_features),  # Dynamic height based on number of features
        width=1200,
        title_text="Features vs Time",
        showlegend=False,
        template='plotly_white',
        margin=dict(t=100, l=50, r=50, b=50)  # Adjust margins
    )

    # Update x-axis title only for the bottom subplot
    fig.update_xaxes(title_text='Time [s]', row=len(selected_features), col=1)

    for i in range(1, len(selected_features) + 1):
        fig.update_yaxes(title_text=selected_features[i-1], row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

# Function to create parameters vs chainage plot
def create_parameters_vs_chainage(df, selected_features, chainage_column, penetration_rates_available=False):
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
              '#9B6B6B', '#E9967A', '#4682B4', '#6B8E23']

    available_features = [f for f in selected_features if f in df.columns]
    
    if not available_features:
        st.warning("None of the selected features are available in the dataset.")
        return

    fig = make_subplots(rows=len(available_features), cols=1,
                        shared_xaxes=True,
                        subplot_titles=available_features,
                        vertical_spacing=0.05)  # Reduced spacing between subplots

    for i, feature in enumerate(available_features, start=1):
        try:
            fig.add_trace(
                go.Scatter(
                    x=df[chainage_column],
                    y=df[feature],
                    mode='lines',
                    name=feature,
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=i,
                col=1
            )
        except Exception as e:
            st.warning(f"Error plotting feature '{feature}': {e}")

    # Update layout with adjusted dimensions
    fig.update_layout(
        height=300 * len(available_features),  # Dynamic height based on number of features
        width=1200,
        title_text='Parameters vs Chainage',
        showlegend=False,
        template='plotly_white',
        margin=dict(t=100, l=50, r=50, b=50)  # Adjust margins
    )

    # Update x-axis title only for the bottom subplot
    fig.update_xaxes(title_text='Chainage [mm]', row=len(available_features), col=1)

    for i in range(1, len(available_features) + 1):
        fig.update_yaxes(title_text=available_features[i-1], row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

# Function to create thrust force plots
def create_thrust_force_plots(df, advance_rate_col):
    try:
        # Identify thrust force column
        thrust_force_col = next((col for col in df.columns 
                           if any(kw in col.lower() for kw in [
                               'thrust force', 'vorschubkraft', 'kraft', 'kraft_max', 
                               'gesamtkraft', 'gesamtkraft_stz', 'gesamtkraft_vtp', 
                               'force'])), 
                          None)
        
        if thrust_force_col is None:
            st.warning("Thrust force column not found in the dataset.")
            return

        # Create subplots
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=("Thrust Force vs Penetration Rate", 
                                         "Thrust Force vs Average Speed", 
                                         "Thrust Force vs Advance Rate"),
                           vertical_spacing=0.1)

        # Plot 1: Thrust Force vs Penetration Rate
        if 'Penetration Rate [mm/rev]' in df.columns:
            mask = df['Penetration Rate [mm/rev]'].notna()
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'Penetration Rate [mm/rev]'], 
                y=df.loc[mask, thrust_force_col], 
                mode='markers', 
                name='vs Penetration Rate', 
                marker=dict(color='blue', size=5)
            ), row=1, col=1)
        else:
            st.warning("Penetration Rate [mm/rev] column not found in the dataset.")

        # Plot 2: Thrust Force vs Average Speed
        if 'Average Speed (mm/min)' in df.columns:
            mask = df['Average Speed (mm/min)'].notna()
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'Average Speed (mm/min)'], 
                y=df.loc[mask, thrust_force_col], 
                mode='markers', 
                name='vs Average Speed',
                marker=dict(color='green', size=5)
            ), row=2, col=1)
        else:
            st.warning("Average Speed (mm/min) column not found in the dataset.")

        # Plot 3: Thrust Force vs Selected Advance Rate
        if advance_rate_col and advance_rate_col in df.columns:
            mask = df[advance_rate_col].notna()
            fig.add_trace(go.Scatter(
                x=df.loc[mask, advance_rate_col], 
                y=df.loc[mask, thrust_force_col], 
                mode='markers', 
                name='vs Advance Rate',
                marker=dict(color='red', size=5)
            ), row=3, col=1)
        else:
            st.warning("Selected advance rate column not available for plotting.")

        # Update layout with improved styling
        fig.update_layout(
            height=1200, 
            width=800, 
            title_text="Thrust Force Relationships",
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels with proper units
        fig.update_xaxes(title_text="Penetration Rate [mm/rev]", row=1, col=1)
        fig.update_xaxes(title_text="Average Speed [mm/min]", row=2, col=1)
        fig.update_xaxes(title_text=advance_rate_col if advance_rate_col else "Advance Rate [mm/min]", row=3, col=1)
        
        for i in range(1, 4):
            fig.update_yaxes(title_text="Thrust Force [kN]", row=i, col=1)

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

# Function to handle chainage filtering and averaging
def handle_chainage_filtering_and_averaging(df, chainage_column, aggregation):
    try:
        st.sidebar.header("Chainage Filtering & Data Averaging")

        # User input for chainage range
        min_chainage = float(df[chainage_column].min())
        max_chainage = float(df[chainage_column].max())
        chainage_range = st.sidebar.slider("Select Chainage Range (mm)", min_chainage, max_chainage, (min_chainage, max_chainage), step=(max_chainage - min_chainage)/100)

        # Filter the DataFrame based on selected range
        filtered_df = df[(df[chainage_column] >= chainage_range[0]) & (df[chainage_column] <= chainage_range[1])]

        st.sidebar.write(f"Selected Chainage Range: {chainage_range[0]} - {chainage_range[1]} mm")

        # Determine aggregation step based on aggregation interval
        if aggregation == '1S':
            # Example: bin every 10 mm
            bins = np.arange(filtered_df[chainage_column].min(), filtered_df[chainage_column].max(), step=10)
        elif aggregation == '1T':
            # Example: bin every 50 mm
            bins = np.arange(filtered_df[chainage_column].min(), filtered_df[chainage_column].max(), step=50)
        else:
            st.warning("Unknown aggregation interval. Skipping aggregation.")
            return filtered_df

        # Bin the chainage and compute mean
        filtered_df['chainage_bin'] = pd.cut(filtered_df[chainage_column], bins=bins, include_lowest=True)
        aggregated_df = filtered_df.groupby('chainage_bin').mean().reset_index()
        aggregated_df[chainage_column] = aggregated_df['chainage_bin'].apply(lambda x: x.mid)

        return aggregated_df
    except Exception as e:
        st.error(f"Error in chainage filtering and averaging: {e}")
        return df

# Function to create path for download link
def generate_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Processed CSV File</a>'
    return href

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

                suggested_working_pressure = suggest_column(df, ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr','SR_Arbdr'])
                suggested_revolution = suggest_column(df, ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz'])
                suggested_advance_rate = suggest_column(df, ['advance rate', 'vortrieb', 'vorschub','VTgeschw','geschw'])

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
                
                df_viz = rename_columns(df.copy(), working_pressure_col, revolution_col, selected_distance, advance_rate_col)

                all_features = df_viz.columns.tolist()
                
                time_column = get_time_column(df_viz)

                options = ['Statistical Summary', 'Parameters vs Chainage', 'Box Plots', 'Violin Plots', 'Thrust Force Plots', 'Correlation Heatmap']
                if time_column:
                    options.extend(['Features vs Time', 'Pressure Distribution'])
                if rock_strength_file:
                    options.append('Rock Strength Comparison')

                selected_option = st.sidebar.radio("Choose visualization", options)

                if selected_option not in ['Pressure Distribution', 'Thrust Force Plots']:
                    default_features = []
                    if 'Calculated torque [kNm]' in all_features:
                        default_features.append('Calculated torque [kNm]')
                    if 'Average Speed (mm/min)' in all_features:
                        default_features.append('Average Speed (mm/min)')
                    if 'Penetration Rate [mm/rev]' in all_features:
                        default_features.append('Penetration Rate [mm/rev]')
                    
                    selected_features = st.sidebar.multiselect(
                        "Select features for analysis",
                        all_features,
                        default=default_features
                    )

                st.subheader(f"Visualization: {selected_option}")

                if selected_option == 'Rock Strength Comparison':
                    rock_df = None
                    if rock_strength_file:
                        rock_strength_data = read_rock_strength_data(rock_strength_file)
                        if rock_strength_data is not None:
                            rock_df = preprocess_rock_strength_data(rock_strength_data)
                            if not rock_df.empty:
                                rock_type = st.sidebar.selectbox("Select Rock Type", rock_df.index)
                                if rock_type and selected_features:
                                    fig = create_rock_strength_comparison_chart(df_viz, rock_df, rock_type, selected_features)
                                    if fig is not None:
                                        st.plotly_chart(fig)
                                else:
                                    st.warning("Please ensure you've selected a rock type and at least one machine parameter for comparison.")
                            else:
                                st.warning("Rock strength data is empty after preprocessing.")
                        else:
                            st.warning("Error processing rock strength data. Please check your file.")
                    else:
                        st.warning("Please upload rock strength data to use this visualization.")
                
                elif selected_option == 'Thrust Force Plots':
                    create_thrust_force_plots(
                        df_viz, 
                        'Advance rate [mm/min]' if advance_rate_col != 'None' else None
                    )
                
                elif selected_option == 'Correlation Heatmap':
                    if selected_features and len(selected_features) > 1:
                        create_correlation_heatmap(df_viz, selected_features)
                    else:
                        st.warning("Please select at least two features for correlation analysis.")
                elif selected_option == 'Statistical Summary':
                    if selected_features:
                        create_statistical_summary(df_viz, selected_features)
                    else:
                        st.warning("Please select features for statistical analysis.")
                elif selected_option == 'Features vs Time' and time_column:
                    if selected_features:
                        # Sampling Rate Handling
                        sampling_rate = st.sidebar.selectbox(
                            "Select Data Sampling Rate",
                            ['Auto Detect', 'Milliseconds', 'Seconds', 'Minutes']
                        )
                        
                        if sampling_rate == 'Auto Detect':
                            # Automatically determine based on time differences
                            df_viz = df_viz.sort_values(by=time_column)
                            df_viz['time_diff'] = df_viz[time_column].diff().fillna(0)

                            average_sampling_interval = df_viz['time_diff'].median()
                            st.sidebar.write(f"Detected average sampling interval: {average_sampling_interval} seconds")
                            
                            if average_sampling_interval < 1:
                                aggregation = '1S'  # Every second
                            else:
                                aggregation = '1T'  # Every minute
                        else:
                            if sampling_rate == 'Milliseconds':
                                aggregation = '1S'  # Aggregate every second
                            elif sampling_rate == 'Seconds':
                                aggregation = '1T'  # Aggregate every minute
                            elif sampling_rate == 'Minutes':
                                aggregation = '1T'  # Aggregate every minute
                            else:
                                aggregation = '1T'  # Default to minute

                        # Aggregate data based on selected interval
                        try:
                            numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
                            numeric_cols.remove(time_column) if time_column in numeric_cols else None
                            df_viz_numeric = df_viz[numeric_cols]
                            df_viz_numeric = df_viz_numeric.resample(aggregation, on=time_column).mean().reset_index()
                            df_viz = pd.merge(df_viz_numeric, df_viz[[time_column]], on=time_column, how='left')
                            st.sidebar.write(f"Data aggregated every {aggregation}")
                        except Exception as e:
                            st.sidebar.error(f"Error parsing time column during aggregation: {e}")
                            df_viz = df_viz.reset_index()

                        create_features_vs_time(df_viz, selected_features, time_column, sampling_rate)
                    else:
                        st.warning("Please select features to visualize over time.")
                elif selected_option == 'Pressure Distribution' and time_column:
                    if working_pressure_col and working_pressure_col != 'None':
                        renamed_pressure_col = 'Working pressure [bar]'
                        create_pressure_distribution_polar_plot(df_viz, renamed_pressure_col, time_column)
                    else:
                        st.warning("Please select a valid working pressure column.")
                elif selected_option == 'Parameters vs Chainage':
                    if selected_features:
                        # **Enhancement 1: Chainage Filtering & Data Averaging Based on Sampling Rate**
                        st.sidebar.header("Chainage Filtering & Data Averaging")

                        # Determine sampling rate from user input or auto detection
                        sampling_rate = st.sidebar.selectbox(
                            "Select Data Sampling Rate for Chainage",
                            ['Auto Detect', 'Milliseconds', 'Seconds', 'Minutes']
                        )
                        
                        if sampling_rate == 'Auto Detect':
                            aggregation = '1S' if st.sidebar.checkbox("Aggregate by Seconds") else '1T'
                        else:
                            if sampling_rate == 'Milliseconds':
                                aggregation = '1S'  # Aggregate every second
                            elif sampling_rate == 'Seconds':
                                aggregation = '1T'  # Aggregate every minute
                            elif sampling_rate == 'Minutes':
                                aggregation = '1T'  # Aggregate every minute
                            else:
                                aggregation = '1S'  # Default to seconds

                        # Chainage Filtering and Averaging
                        df_viz = handle_chainage_filtering_and_averaging(df_viz, 'Chainage [mm]', aggregation)

                        create_parameters_vs_chainage(df_viz, selected_features, 'Chainage [mm]', penetration_rates_available=('Sensor-based Penetration Rate' in df_viz.columns))
                    else:
                        st.warning("Please select features to visualize against chainage.")
                elif selected_option == 'Box Plots':
                    if selected_features:
                        create_multi_axis_box_plots(df_viz, selected_features)
                    else:
                        st.warning("Please select features for box plot analysis.")
                elif selected_option == 'Violin Plots':
                    if selected_features:
                        create_multi_axis_violin_plots(df_viz, selected_features)
                    else:
                        st.warning("Please select features for violin plot analysis.")

                # Download processed data
                if st.sidebar.button("Download Processed Data"):
                    download_link = generate_download_link(df_viz, "processed_data.csv")
                    st.sidebar.markdown(download_link, unsafe_allow_html=True)

            else:
                st.error("Error loading the data. Please check your file format.")
    except Exception as e:
        st.error(f"An unexpected error occurred in the main function: {str(e)}")

    st.markdown("---")
    st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
    st.markdown("**Created by Kursat Kilic - Geotechnical Digitalization**")

def suggest_column(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                return col
    return None

if __name__ == "__main__":
    main()
