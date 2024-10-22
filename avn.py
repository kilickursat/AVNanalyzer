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

# Advanced rate calculation function
def calculate_advance_rate_and_stats(df, distance_column, time_column):
    try:
        if not all(col in df.columns for col in [distance_column, time_column]):
            raise ValueError(f"Required columns not found in DataFrame")
            
        if len(df) > 1:
            weg = round(df[distance_column].max() - df[distance_column].min(), 2)
            zeit = round((df[time_column].max() - df[time_column].min()).total_seconds() / 60, 2)  # Convert to minutes
        else:
            weg = round(df[distance_column].iloc[0], 2)
            zeit = round(df[time_column].iloc[0].timestamp() / 60, 2)  # Convert to minutes if single entry
            
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
        st.error(f"Error calculating penetration rate: {e}")
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
            if result:
                df['Average Speed (mm/min)'] = average_speed
            
            if revolution_col is not None:
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
    distance_keywords = ['distance', 'length', 'travel', 'chainage', 'tunnellänge neu', 'tunnellänge', 'weg_mm_z', 'vtp_weg']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in distance_keywords)]

def get_time_column(df):
    time_keywords = ['relativzeit', 'relative time', 'time', 'datum', 'date', 'zeit', 'timestamp', 'Relative Time', 'Relativzeit']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
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

# Updated function to create Features vs Time plot
def create_features_vs_time(df, selected_features, time_column, sampling_rate):
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
                        vertical_spacing=0.1)  # Increased spacing between subplots

    for i, feature in enumerate(available_features, start=1):
        try:
            y_data = df[feature]
            feature_name = feature

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
            
            # Plot Penetration Rates if available
            if feature == 'Penetration Rate [mm/rev]' and penetration_rates_available:
                fig.add_trace(
                    go.Scatter(
                        x=df[chainage_column],
                        y=df['Penetration Rate [mm/rev]'],
                        mode='lines',
                        name='Calculated Penetration Rate',
                        line=dict(color='blue', dash='dash')
                    ),
                    row=i,
                    col=1
                )
            elif feature == 'Sensor-based Penetration Rate' and penetration_rates_available:
                fig.add_trace(
                    go.Scatter(
                        x=df[chainage_column],
                        y=df['Sensor-based Penetration Rate'],
                        mode='lines',
                        name='Sensor-based Penetration Rate',
                        line=dict(color='green', dash='dot')
                    ),
                    row=i,
                    col=1
                )

            # Update y-axis titles with more space
            fig.update_yaxes(
                title_text=feature_name, 
                row=i, 
                col=1,
                title_standoff=40  # Increased standoff to prevent overlap
            )
        except Exception as e:
            st.warning(f"Error plotting feature '{feature}': {e}")

    # Update layout with adjusted dimensions
    fig.update_layout(
        height=300 * len(available_features),  # Dynamic height based on number of features
        width=1200,
        title_text=f'Parameters vs Chainage',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100, l=150, r=50, b=50)  # Increased left margin for y-axis labels
    )

    # Update x-axis title only for the bottom subplot
    fig.update_xaxes(title_text='Chainage [mm]', row=len(available_features), col=1)

    st.plotly_chart(fig, use_container_width=True)

# Updated function to create multi-axis box plots with additional features
def create_multi_axis_box_plots(df, selected_features):
    if not selected_features:
        st.warning("Please select at least one feature for the box plots.")
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

def handle_chainage_filtering_and_averaging(df, chainage_column, aggregation):
    try:
        st.sidebar.header("Chainage Filtering")

        # User input for chainage range
        min_chainage = float(df[chainage_column].min())
        max_chainage = float(df[chainage_column].max())
        chainage_range = st.sidebar.slider("Select Chainage Range (mm)", min_chainage, max_chainage, (min_chainage, max_chainage), step=(max_chainage - min_chainage)/100)

        # Filter the DataFrame based on selected range
        filtered_df = df[(df[chainage_column] >= chainage_range[0]) & (df[chainage_column] <= chainage_range[1])]

        st.sidebar.write(f"Selected Chainage Range: {chainage_range[0]} - {chainage_range[1]} mm")

        # Aggregate data based on aggregation interval
        if aggregation in ['1S', '5S', '10S', '30S']:
            # Assuming chainage increases with time, sort by chainage
            filtered_df = filtered_df.sort_values(by=chainage_column)
            # Resample based on chainage by grouping
            # Since chainage isn't datetime, binning is necessary
            bins = np.arange(filtered_df[chainage_column].min(), filtered_df[chainage_column].max(), step=10)  # Example step of 10 mm
            filtered_df['chainage_bin'] = pd.cut(filtered_df[chainage_column], bins=bins, include_lowest=True)
            aggregated_df = filtered_df.groupby('chainage_bin').mean().reset_index()
            aggregated_df[chainage_column] = aggregated_df['chainage_bin'].apply(lambda x: x.mid)
            return aggregated_df
        elif aggregation in ['1T', '5T', '10T', '30T']:
            # For chainage, binning is similar
            filtered_df = filtered_df.sort_values(by=chainage_column)
            bins = np.arange(filtered_df[chainage_column].min(), filtered_df[chainage_column].max(), step=50)  # Example step of 50 mm
            filtered_df['chainage_bin'] = pd.cut(filtered_df[chainage_column], bins=bins, include_lowest=True)
            aggregated_df = filtered_df.groupby('chainage_bin').mean().reset_index()
            aggregated_df[chainage_column] = aggregated_df['chainage_bin'].apply(lambda x: x.mid)
            return aggregated_df
        else:
            st.warning("Unknown aggregation interval. Skipping aggregation.")
            return filtered_df
    except Exception as e:
        st.error(f"Error in chainage filtering and averaging: {e}")
        return df

# Updated function to create thrust force plots
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

def main():
    try:
        set_background_color()
        add_logo()

        st.title("Herrenknecht Hard Rock Data Analysis App")

        st.sidebar.header("Data Upload & Analysis")

        # File upload
        uploaded_file = st.sidebar.file_uploader("Machine Data (CSV/Excel)", type=['csv', 'xlsx'], key='file_uploader_1')
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                try:
                    try:
                    df = pd.read_csv(uploaded_file, error_bad_lines=False)
                except pd.errors.ParserError as e:
                    st.error(f"Error reading CSV file: {e}")
                    return
                except pd.errors.ParserError as e:
                    st.error(f"Error reading CSV file: {e}")
                    return
            elif uploaded_file.name.endswith('.xlsx'):
                try:
                    df = pd.read_excel(uploaded_file)
                except ValueError as e:
                    st.error(f"Error reading Excel file: {e}")
                    return
            st.write("Data preview:", df.head())

            # 1. Filtering Data based on Chainage
            st.sidebar.header("Data Filtering Options")
            chainage_filter = st.sidebar.checkbox("Enable Chainage Filtering")
            if chainage_filter:
                filter_type = st.sidebar.selectbox("Filter Type", ["Average over Length", "Select Chainage Section"])
                if filter_type == "Average over Length":
                    length_unit = st.sidebar.selectbox("Select Unit", ["mm", "cm", "m"])
                    length_value = st.sidebar.number_input("Enter Length for Averaging", min_value=0.01, step=0.01)
                    # Convert all to meters for simplicity
                    if length_unit == "mm":
                        length_value /= 1000
                    elif length_unit == "cm":
                        length_value /= 100
                    # Average data based on chainage length provided by the user
                    df['chainage'] = df['chainage'] // length_value * length_value
                    df_avg = df.groupby('chainage').mean().reset_index()
                    st.write("Filtered/Averaged Data:", df_avg.head())
                elif filter_type == "Select Chainage Section":
                    start_chainage = st.sidebar.number_input("Enter Start Chainage (m)", min_value=0.0, step=0.1)
                    end_chainage = st.sidebar.number_input("Enter End Chainage (m)", min_value=0.0, step=0.1)
                    resolution = st.sidebar.number_input("Resolution for Section (m)", min_value=0.01, step=0.01)
                    df_filtered = df[(df['chainage'] >= start_chainage) & (df['chainage'] <= end_chainage)]
                    st.write("Filtered Data by Chainage Section:", df_filtered.head())
            else:
                df_avg = df  # Keep the original dataframe as is

            # 2. Input for Cutting Rings
            st.sidebar.header("Thrust Force Calculation")
            thrust_force = st.sidebar.number_input("Enter Thrust Force", min_value=0.0)
            number_cutting_rings = st.sidebar.number_input("Number of Cutting Rings", min_value=1)
            cutting_ring_thrust = thrust_force / number_cutting_rings if number_cutting_rings > 0 else 0
            st.sidebar.write(f"Thrust Force per Cutting Ring: {cutting_ring_thrust}")

            # 3. Sensor and Calculated Penetrations
            st.header("Penetration Rates")
            df['sensor_penetration'] = df['penetration_sensor']  # Assuming 'penetration_sensor' exists
            df['calculated_penetration'] = df['penetration_calculated']  # Assuming 'penetration_calculated' exists
            st.write(df[['chainage', 'sensor_penetration', 'calculated_penetration']])

            # 4. Thrust Force vs Penetration Rate Plot
            st.header("Thrust Force vs Calculated Penetration Rate")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='calculated_penetration', y='thrust_force', data=df, hue='chainage', palette='viridis', alpha=0.6)
            plt.xlabel("Calculated Penetration Rate")
            plt.ylabel("Thrust Force")
            plt.title("Data Distribution of Thrust Force vs Calculated Penetration Rate")
            st.pyplot(plt)

            # 5. Parameters vs Chainage and Features vs Time Plots
            st.header("Parameters vs Chainage and Features vs Time")
            chainage_or_time = st.sidebar.selectbox("Select X Axis for Plots", ["Chainage", "Time"])
            param_cols = st.multiselect("Select Parameters to Plot", df.columns.tolist())
            if chainage_or_time == "Chainage":
                fig, axes = plt.subplots(len(param_cols), 1, figsize=(10, 5 * len(param_cols)))
                if len(param_cols) == 1:
                    axes = [axes]  # Convert to list for consistency
                for ax, param in zip(axes, param_cols):
                    sns.lineplot(x='chainage', y=param, data=df_avg, ax=ax)
                    ax.set_xlabel("Chainage")
                    ax.set_ylabel(param)
                st.pyplot(fig)
            elif chainage_or_time == "Time":
                df_avg['time'] = pd.to_datetime(df_avg['time'], errors='coerce')
                fig, axes = plt.subplots(len(param_cols), 1, figsize=(10, 5 * len(param_cols)))
                if len(param_cols) == 1:
                    axes = [axes]
                for ax, param in zip(axes, param_cols):
                    sns.lineplot(x='time', y=param, data=df_avg, ax=ax)
                    ax.set_xlabel("Time")
                    ax.set_ylabel(param)
                st.pyplot(fig)

        else:
            st.write("Please upload a CSV or Excel file to proceed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

