import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
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
                padding-top: 120px;  /* Consistent padding */
            }
            [data-testid="stSidebar"]::before {
                content: "";
                margin-bottom: 20px;  /* Consistent margin */
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
# Function to calculate derived features and fix penetration rate issue
def calculate_derived_features(df, pressure_column, revolution_column, n1, torque_constant, distance_column, time_column):
    try:
        # Ensure numeric conversion for critical columns
        df[pressure_column] = pd.to_numeric(df[pressure_column], errors='coerce')
        df[revolution_column] = pd.to_numeric(df[revolution_column], errors='coerce')
        df[distance_column] = pd.to_numeric(df[distance_column], errors='coerce')

        # Example derived feature calculations
        df['Penetration Rate [mm/rev]'] = df[distance_column] / df[revolution_column]
        df['Torque [kNm]'] = df[pressure_column] * torque_constant

        # Handling NaN values in Penetration Rate
        df['Penetration Rate [mm/rev]'].fillna(0, inplace=True)

        # Normalize time for proper plotting
        df['normalized_time'] = (df[time_column] - df[time_column].min()) / (df[time_column].max() - df[time_column].min()) * 360
        return df
    except Exception as e:
        st.error(f"An error occurred in feature calculation: {str(e)}")
        return None

# Function to handle Chainage mid aggregation
def aggregate_chainage(df):
    try:
        df['Chainage_mid'] = pd.to_numeric(df['Chainage_mid'], errors='coerce')
        df['Chainage_mid'].fillna(df['Chainage_mid'].mean(), inplace=True)
        return df
    except Exception as e:
        st.error(f"Aggregation error: {str(e)}")
        return None
# Function to fix Features vs Time memory issue (sampling large datasets)
def downsample_large_datasets(df):
    if len(df) > 10000:
        return df.iloc[::10, :]
    return df
    
# Function for Parameters vs Chainage plot
def plot_parameters_vs_chainage(df, parameter_columns, chainage_column):
    try:
        df = aggregate_chainage(df)
        fig = go.Figure()
        for parameter in parameter_columns:
            fig.add_trace(go.Scatter(x=df[chainage_column], y=df[parameter], mode='lines', name=parameter))
        fig.update_layout(
            title='Parameters vs Chainage',
            xaxis_title='Chainage Mid',
            yaxis_title='Parameter Value',
            showlegend=True
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting Parameters vs Chainage: {str(e)}")

# Main function to run Streamlit app and provide visualization options
def main():
    st.title('Data Visualization Tool')

    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

            st.write("Data Loaded Successfully")
            st.write(df.head())

            working_pressure_col = st.sidebar.text_input("Working Pressure Column", 'WorkingPressure')
            revolution_col = st.sidebar.text_input("Revolution Column", 'Revolutions')
            distance_col = st.sidebar.text_input("Distance Column", 'Distance')
            time_column = st.sidebar.text_input("Time Column", 'Time')

            torque_constant = st.sidebar.number_input("Torque Constant", value=1.0)

            df = calculate_derived_features(df, working_pressure_col, revolution_col, 1, torque_constant, distance_col, time_column)
            if df is None:
                return

            visualization_options = ['Parameters vs Chainage', 'Thrust Force Plots', 'Features vs Time']

            selected_visualization = st.sidebar.selectbox("Choose Visualization", visualization_options)


# Function to plot Features vs Time
def plot_features_vs_time(df, selected_features, time_column):
    try:
        df = downsample_large_datasets(df)
        fig = go.Figure()
        for feature in selected_features:
            fig.add_trace(go.Scatter(x=df[time_column], y=df[feature], mode='lines', name=feature))
        fig.update_layout(
            title='Features vs Time',
            xaxis_title='Time',
            yaxis_title='Feature Value',
            showlegend=True
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting: {str(e)}")

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
    distance_keywords = ['distance', 'length', 'travel', 'chainage', 'tunnellänge neu', 'tunnellänge', 'weg_mm_z', 'vtp_weg']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in distance_keywords)]

# Modified get_time_column function to handle relative time correctly
def suggest_column(df, keywords):
    """Helper function to suggest columns based on keywords"""
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                return col
    return None

def get_time_column(df):
    """Enhanced function to identify time column"""
    time_keywords = ['relativzeit', 'relative time', 'time', 'datum', 'date', 
                    'zeit', 'timestamp', 'Relative Time', 'Relativzeit']
    
    # First try to find datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        return datetime_cols[0]
    
    # Then look for columns with time-related keywords
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            try:
                # Try to convert to datetime
                pd.to_datetime(df[col], errors='raise')
                return col
            except:
                # If conversion fails, it might be numeric relative time
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
    
    return None

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
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True, engine='python')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported rock strength file format")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading rock strength data: {e}")
        return None

# Function to preprocess the rock strength data
def preprocess_rock_strength_data(df):
    try:
        # Assuming 'Probenbezeichnung' contains rock type information
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

# Updated function to create correlation heatmap with dynamic input
def create_correlation_heatmap(df, selected_features):
    if len(selected_features) < 2:
        st.warning("Please select at least two features for the correlation heatmap.")
        return

    # Only use the features that the user has explicitly selected and are numeric
    available_features = [f for f in selected_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    if len(available_features) < 2:
        st.warning("Please select at least two valid numeric features for the correlation heatmap.")
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
    # Automatically include 'Penetration Rate [mm/rev]' if present
    if 'Penetration Rate [mm/rev]' in df.columns and 'Penetration Rate [mm/rev]' not in selected_features:
        selected_features.append('Penetration Rate [mm/rev]')
    
    if not selected_features:
        st.warning("Please select at least one feature for the statistical summary.")
        return

    # Only include numeric features
    selected_features = [f for f in selected_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not selected_features:
        st.warning("No numeric features selected for statistical summary.")
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

# Updated function to create Features vs Time plot
def create_features_vs_time(df, selected_features, time_column, sampling_rate):
    if not selected_features:
        st.warning("Please select at least one feature for the time series plot.")
        return

    # Only include numeric features
    selected_features = [f for f in selected_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not selected_features:
        st.warning("No numeric features selected for the time series plot.")
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
                x=df['Time_mid'] if 'Time_mid' in df.columns else df[time_column],
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
        # Check if the pressure column exists, if not, try to find a similar column
        if pressure_column not in df.columns:
            potential_columns = [col for col in df.columns if 'pressure' in col.lower() or 'druck' in col.lower()]
            if potential_columns:
                pressure_column = potential_columns[0]
                st.warning(f"Original pressure column not found. Using '{pressure_column}' instead.")
            else:
                st.error(f"Could not find a suitable pressure column. Please check your data.")
                return

        df[pressure_column] = pd.to_numeric(df[pressure_column], errors='coerce')

        # Normalize time to 360 degrees
        if pd.api.types.is_numeric_dtype(df[time_column]):
            df['normalized_time'] = (df[time_column] - df[time_column].min()) / (df[time_column].max() - df[time_column].min()) * 360
        else:
            # Assuming datetime, convert to seconds for normalization
            df['normalized_time'] = (df[time_column] - df[time_column].min()).dt.total_seconds()
            df['normalized_time'] = (df['normalized_time'] / df['normalized_time'].max()) * 360

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=df[pressure_column],
            theta=df['normalized_time'],
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Pressure'
        ))

        max_pressure = df[pressure_column].max()
        if pd.isna(max_pressure) or max_pressure == 0:
            max_pressure = 1  # Avoid zero range

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

# Updated function to create multi-axis box plots with additional features
def create_multi_axis_box_plots(df, selected_features):
    if not selected_features:
        st.warning("Please select at least one feature for the box plots.")
        return

    # Only include numeric features
    selected_features = [f for f in selected_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not selected_features:
        st.warning("No numeric features selected for box plots.")
        return

    try:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])
        colors = ['#0000cd', '#6495ed', '#4b0082', '#ff00ff']  # Corresponding colors

        for i, feature in enumerate(selected_features):
            fig.add_trace(go.Box(y=df[feature], name=feature, marker_color=colors[i % len(colors)]))

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

    # Only include numeric features
    selected_features = [f for f in selected_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not selected_features:
        st.warning("No numeric features selected for violin plots.")
        return

    try:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])
        colors = ['#0000cd', '#6495ed', '#4b0082', '#ff00ff']  # Corresponding colors

        for i, feature in enumerate(selected_features):
            fig.add_trace(go.Violin(y=df[feature], name=feature, box_visible=True, meanline_visible=True, fillcolor=colors[i % len(colors)]))

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

# Updated function to handle sampling aggregation for time-based data
def handle_sampling_aggregation(df, time_column, aggregation, agg_type='time'):
    try:
        df = df.sort_values(by=time_column)
        if agg_type == 'time':
            if pd.api.types.is_numeric_dtype(df[time_column]):
                bins = np.arange(df[time_column].min(), df[time_column].max() + float(aggregation.rstrip('S').rstrip('T')), float(aggregation.rstrip('S').rstrip('T')))
                df['time_bin'] = pd.cut(df[time_column], bins=bins)
                df_agg = df.groupby('time_bin').mean().reset_index()
                df_agg['Time_mid'] = df_agg['time_bin'].apply(lambda x: x.mid if pd.notnull(x) else np.nan)
                return df_agg
            else:
                # For datetime, resampling is handled outside
                return df
        elif agg_type == 'chainage':
            # Implement chainage-based aggregation if needed
            bins = np.arange(df[time_column].min(), df[time_column].max() + float(aggregation.rstrip('S').rstrip('T')), float(aggregation.rstrip('S').rstrip('T')))
            df['chainage_bin'] = pd.cut(df[time_column], bins=bins)
            df_agg = df.groupby('chainage_bin').mean().reset_index()
            df_agg['Chainage_mid'] = df_agg['chainage_bin'].apply(lambda x: x.mid if pd.notnull(x) else np.nan)
            return df_agg
        else:
            return df
    except Exception as e:
        st.error(f"Error in sampling aggregation: {e}")
        return df

# Function for Thrust Force Plots
def plot_thrust_force(df, selected_features):
    try:
        fig = go.Figure()
        for feature in selected_features:
            fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name=feature))
        fig.update_layout(
            title='Thrust Force Plots',
            xaxis_title='Index',
            yaxis_title='Thrust Force',
            showlegend=True
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting Thrust Force: {str(e)}")


        # Display correlation statistics
        st.subheader("Correlation Statistics")
        corr_stats = pd.DataFrame(columns=['Parameter', 'Correlation with Thrust Force', 'R-squared'])
        
        for feature in plot_features:
            if feature in df.columns:
                mask = df[feature].notna() & df[thrust_force_col].notna()
                if mask.sum() > 1 and df.loc[mask, feature].nunique() > 1:
                    correlation = df.loc[mask, [feature, thrust_force_col]].corr().iloc[0, 1]
                    _, _, r_value, _, _ = stats.linregress(df.loc[mask, feature], df.loc[mask, thrust_force_col])
                    corr_stats = corr_stats.append({
                        'Parameter': feature,
                        'Correlation with Thrust Force': correlation,
                        'R-squared': r_value**2
                    }, ignore_index=True)
                else:
                    st.warning(f"Insufficient data to calculate correlation for '{feature}'.")

        if not corr_stats.empty:
            st.dataframe(corr_stats.style.format({
                'Correlation with Thrust Force': '{:.3f}',
                'R-squared': '{:.3f}'
            }))
        else:
            st.info("No correlation statistics available.")

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

# Main function to run Streamlit app and provide visualization options
def main():
    st.title('Data Visualization Tool')

    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

            st.write("Data Loaded Successfully")
            st.write(df.head())

            working_pressure_col = st.sidebar.text_input("Working Pressure Column", 'WorkingPressure')
            revolution_col = st.sidebar.text_input("Revolution Column", 'Revolutions')
            distance_col = st.sidebar.text_input("Distance Column", 'Distance')
            time_column = st.sidebar.text_input("Time Column", 'Time')

            torque_constant = st.sidebar.number_input("Torque Constant", value=1.0)

            df = calculate_derived_features(df, working_pressure_col, revolution_col, 1, torque_constant, distance_col, time_column)
            if df is None:
                return

            visualization_options = ['Parameters vs Chainage', 'Thrust Force Plots', 'Features vs Time']

            selected_visualization = st.sidebar.selectbox("Choose Visualization", visualization_options)

                if selected_visualization == 'Parameters vs Chainage':
                param_columns = st.sidebar.multiselect("Select Parameters", df.columns)
                chainage_column = st.sidebar.text_input("Chainage Column", 'Chainage_mid')
                plot_parameters_vs_chainage(df, param_columns, chainage_column)

            elif selected_visualization == 'Thrust Force Plots':
                selected_features = st.sidebar.multiselect("Select Features for Thrust Force", df.columns)
                if selected_features:
                    plot_thrust_force(df, selected_features)
                else:
                    st.error("Please select features for plotting Thrust Force")

            elif selected_visualization == 'Features vs Time':
                selected_features = st.sidebar.multiselect("Select Features for Time Plot", df.columns)
                if selected_features:
                    plot_features_vs_time(df, selected_features, time_column)
                else:
                    st.error("Please select features for plotting Time")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")


    st.markdown("---")
    st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
    st.markdown("**Created by Kursat Kilic - Geotechnical Digitalization**")

# Additional utility functions (if any) can go here
def rename_columns(df, pressure_column, revolution_column, distance_column, advance_rate_column):
    # Example renaming logic, based on actual requirements
    df.rename(columns={
        pressure_column: 'Working Pressure',
        revolution_column: 'Revolutions',
        distance_column: 'Distance',
        advance_rate_column: 'Advance Rate'
    }, inplace=True)
    return df

if __name__ == "__main__":
    main()

