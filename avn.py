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
def calculate_derived_features(df, working_pressure_col, revolution_col, n1, torque_constant, selected_distance, time_col):
    try:
        # Calculate torque
        if working_pressure_col and revolution_col and revolution_col != 'None':
            df[working_pressure_col] = pd.to_numeric(df[working_pressure_col], errors='coerce')
            df[revolution_col] = pd.to_numeric(df[revolution_col], errors='coerce')

            # Calculate torque
            df['Calculated torque [kNm]'] = df.apply(
                lambda row: calculate_torque(row[working_pressure_col], torque_constant, row[revolution_col], n1)
                if pd.notna(row[working_pressure_col]) and pd.notna(row[revolution_col]) else np.nan,
                axis=1
            )

        # Calculate penetration rate for each row as Advance rate [mm/min] / Revolution [rpm]
        if 'Advance rate [mm/min]' in df.columns and revolution_col in df.columns:
            df['Penetration Rate [mm/rev]'] = df.apply(
                lambda row: row['Advance rate [mm/min]'] / row[revolution_col] 
                if pd.notna(row['Advance rate [mm/min]']) and pd.notna(row[revolution_col]) and row[revolution_col] != 0 
                else np.nan,
                axis=1
            )

        # Calculate average speed (mm/min)
        if time_col and selected_distance:
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                total_time = (df[time_col].max() - df[time_col].min()).total_seconds() / 60  # Convert to minutes
            else:
                # Assuming time_col is in milliseconds
                total_time = (df[time_col].max() - df[time_col].min()) / 60000  # Convert milliseconds to minutes
            
            total_distance = df[selected_distance].max() - df[selected_distance].min()
            average_speed = total_distance / total_time if total_time > 0 else np.nan
            df['Average Speed (mm/min)'] = average_speed
        else:
            df['Average Speed (mm/min)'] = np.nan

        return df

    except Exception as e:
        st.error(f"Error calculating derived features: {e}")
        return df

def create_parameters_vs_chainage(df, selected_features, chainage_column, penetration_rates_available=False, aggregation=None):
    if not selected_features:
        st.warning("Please select at least one feature for the chainage plot.")
        return

    # Ensure the chainage column exists
    if chainage_column not in df.columns:
        st.error(f"Chainage column '{chainage_column}' not found in the dataset.")
        return

    # Ensure 'Chainage_mid' exists after aggregation
    if 'Chainage_mid' not in df.columns:
        st.error("Aggregated dataframe does not contain 'Chainage_mid' column.")
        return

    # Sort the data by chainage_mid
    df = df.sort_values(by='Chainage_mid')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5',
              '#9B6B6B', '#E9967A', '#4682B4', '#6B8E23']

    available_features = [f for f in selected_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    if not available_features:
        st.warning("None of the selected numeric features are available in the dataset.")
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
                    x=df['Chainage_mid'],
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
                        x=df['Chainage_mid'],
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
                        x=df['Chainage_mid'],
                        y=df['Penetration Rate [mm/rev]'],
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
        title_text='Parameters vs Chainage',
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

# Updated function to handle chainage filtering and averaging with aggregation
def handle_chainage_filtering_and_averaging(df, chainage_column, aggregation_method='auto'):
    try:
        # Determine aggregation interval based on user selection or auto detection
        if aggregation_method == 'auto':
            bin_size = (df[chainage_column].max() - df[chainage_column].min()) / 1000  # Adjust bin size as needed
        else:
            # Convert aggregation_method to numeric bin size if possible
            try:
                # If aggregation is time-based, interpret 'S' as chainage units (e.g., mm)
                # Assuming '1S' = 1 unit, '5S' = 5 units, etc.
                if aggregation_method.endswith('S'):
                    bin_size = float(aggregation_method.rstrip('S'))
                elif aggregation_method.endswith('T'):
                    bin_size = float(aggregation_method.rstrip('T')) * 10  # Example conversion
                else:
                    bin_size = (df[chainage_column].max() - df[chainage_column].min()) / 1000  # Default bin size
            except ValueError:
                bin_size = (df[chainage_column].max() - df[chainage_column].min()) / 1000  # Default bin size

        # Create bins for chainage
        df = df.sort_values(by=chainage_column)
        bins = np.arange(df[chainage_column].min(), df[chainage_column].max() + bin_size, bin_size)
        df['chainage_bin'] = pd.cut(df[chainage_column], bins=bins)

        # Aggregate data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_agg = df.groupby('chainage_bin')[numeric_columns].agg(['mean', 'std', 'min', 'max']).reset_index()

        # Flatten MultiIndex columns
        df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]

        # Extract the mid-point of each bin for plotting
        df_agg['Chainage_mid'] = df['chainage_bin'].cat.categories.mid

        return df_agg

    except Exception as e:
        st.error(f"Error in chainage filtering and averaging: {e}")
        return df

# Updated function to create thrust force plots
def create_thrust_force_plots(df, advance_rate_col):
    try:
        # Let user select thrust force column
        thrust_force_cols = [col for col in df.columns 
                           if any(kw in col.lower() for kw in [
                               'thrust force', 'vorschubkraft', 'kraft', 'kraft_max', 
                               'gesamtkraft', 'gesamtkraft_stz', 'gesamtkraft_vtp', 
                               'force'])]
        
        if not thrust_force_cols:
            st.warning("No thrust force columns found in the dataset.")
            return

        thrust_force_col = st.selectbox("Select Thrust Force Column", thrust_force_cols)
        
        # Convert thrust force to numeric if needed
        df[thrust_force_col] = pd.to_numeric(df[thrust_force_col], errors='coerce')

        # Create subplots using raw data (no averaging)
        fig = make_subplots(rows=2, cols=1,  # Changed to 2 plots excluding Average Speed
                           subplot_titles=("Thrust Force vs Penetration Rate", 
                                         "Thrust Force vs Advance Rate"),
                           vertical_spacing=0.1)

        # Plot 1: Thrust Force vs Penetration Rate
        if 'Penetration Rate [mm/rev]' in df.columns:
            mask = df['Penetration Rate [mm/rev]'].notna() & df[thrust_force_col].notna()
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, 'Penetration Rate [mm/rev]'],
                    y=df.loc[mask, thrust_force_col],
                    mode='markers',
                    name='Penetration Rate',
                    marker=dict(
                        color='blue',
                        size=5,
                        opacity=0.6
                    )
                ),
                row=1, col=1
            )

        # Plot 2: Thrust Force vs Advance Rate
        if advance_rate_col and advance_rate_col in df.columns:
            mask = df[advance_rate_col].notna() & df[thrust_force_col].notna()
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, advance_rate_col],
                    y=df.loc[mask, thrust_force_col],
                    mode='markers',
                    name='Advance Rate',
                    marker=dict(
                        color='red',
                        size=5,
                        opacity=0.6
                    )
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            height=800,  # Adjusted height for two plots
            width=800,
            title_text=f"Thrust Force Relationships - {thrust_force_col}",
            showlegend=True,
            template='plotly_white'
        )

        # Update axes labels
        fig.update_xaxes(title_text="Penetration Rate [mm/rev]", row=1, col=1)
        fig.update_xaxes(title_text="Advance Rate [mm/min]", row=2, col=1)

        for i in range(1, 3):
            fig.update_yaxes(title_text=f"{thrust_force_col} [kN]", row=i, col=1)

        # Add trend lines
        for i, (x_col, row) in enumerate([
            ('Penetration Rate [mm/rev]', 1),
            (advance_rate_col, 2)
        ]):
            if x_col in df.columns:
                mask = df[x_col].notna() & df[thrust_force_col].notna()
                x = df.loc[mask, x_col]
                y = df.loc[mask, thrust_force_col]
                
                if len(x) > 1 and x.nunique() > 1:  # Check for at least 2 unique X values
                    slope, intercept, r_value, _, _ = stats.linregress(x, y)
                    x_range = np.linspace(x.min(), x.max(), 100)
                    y_range = slope * x_range + intercept
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode='lines',
                            name=f'Trend (R² = {r_value**2:.3f})',
                            line=dict(color='black', dash='dash')
                        ),
                        row=row, col=1
                    )
                else:
                    st.warning(f"Cannot perform linear regression for '{x_col}' as all X values are identical or insufficient data.")

        st.plotly_chart(fig)

        # Display correlation statistics
        st.subheader("Correlation Statistics")
        corr_stats = pd.DataFrame(columns=['Parameter', 'Correlation with Thrust Force', 'R-squared'])
        
        for param in ['Penetration Rate [mm/rev]', 'Advance Rate [mm/min]']:
            if param in df.columns:
                mask = df[param].notna() & df[thrust_force_col].notna()
                if mask.sum() > 1 and df.loc[mask, param].nunique() > 1:
                    correlation = df.loc[mask, [param, thrust_force_col]].corr().iloc[0, 1]
                    _, _, r_value, _, _ = stats.linregress(df.loc[mask, param], df.loc[mask, thrust_force_col])
                    corr_stats = corr_stats.append({
                        'Parameter': param,
                        'Correlation with Thrust Force': correlation,
                        'R-squared': r_value**2
                    }, ignore_index=True)
                else:
                    st.warning(f"Insufficient data to calculate correlation for '{param}'.")

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

                time_column = get_time_column(df)

                if working_pressure_col != 'None' and revolution_col != 'None' and time_column:
                    df = calculate_derived_features(df, working_pressure_col, revolution_col, n1, torque_constant, selected_distance, time_column)
                
                df_viz = rename_columns(df.copy(), working_pressure_col, revolution_col, selected_distance, advance_rate_col)

                all_features = df_viz.columns.tolist()
                
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
                    if 'Average Speed (mm/min)' in all_features and not df_viz['Average Speed (mm/min)'].isna().all():
                        default_features.append('Average Speed (mm/min)')
                    if 'Penetration Rate [mm/rev]' in all_features and not df_viz['Penetration Rate [mm/rev]'].isna().all():
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
                            if rock_df is not None and not rock_df.empty:
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
                        # Determine sampling rate based on user input or auto detection
                        sampling_rate = st.sidebar.selectbox(
                            "Select Data Sampling Rate",
                            ['Auto Detect', 'Milliseconds', 'Seconds', 'Minutes']
                        )
                        
                        if sampling_rate == 'Auto Detect':
                            # Automatically determine based on time differences
                            df_viz = df_viz.sort_values(by=time_column)
                            if pd.api.types.is_numeric_dtype(df_viz[time_column]):
                                average_sampling_interval = df_viz[time_column].diff().median()
                                st.sidebar.write(f"Detected average sampling interval: {average_sampling_interval} units")
                                
                                if average_sampling_interval < 1:
                                    aggregation = '1S'  # Every second
                                else:
                                    aggregation = '1T'  # Every minute
                            else:
                                df_viz['time_diff'] = df_viz[time_column].diff().dt.total_seconds()
                                average_sampling_interval = df_viz['time_diff'].median()
                                st.sidebar.write(f"Detected average sampling interval: {average_sampling_interval} seconds")
                                
                                if average_sampling_interval < 60:
                                    aggregation = '1S'  # Every second
                                else:
                                    aggregation = '1T'  # Every minute
                        else:
                            if sampling_rate == 'Milliseconds':
                                aggregation = '1S'  # Aggregate every second
                            elif sampling_rate == 'Seconds':
                                aggregation = '1S'  # Aggregate every second
                            elif sampling_rate == 'Minutes':
                                aggregation = '1T'  # Aggregate every minute
                            else:
                                aggregation = '1S'  # Default to second

                        # Aggregate data based on selected interval
                        if aggregation.startswith('1S') or aggregation.startswith('5S') or aggregation.startswith('10S') or aggregation.startswith('30S') or \
                           aggregation.startswith('1T') or aggregation.startswith('5T') or aggregation.startswith('10T') or aggregation.startswith('30T'):
                            if pd.api.types.is_numeric_dtype(df_viz[time_column]):
                                # For relative time (numeric), binning based on aggregation
                                aggregated_df = handle_chainage_filtering_and_averaging(df_viz, time_column, aggregation)
                                if 'Chainage_mid' not in aggregated_df.columns:
                                    st.error("Aggregation failed to create 'Chainage_mid' column.")
                                    aggregated_df = df_viz
                            else:
                                # For datetime, resampling
                                df_viz = df_viz.set_index(time_column)
                                if aggregation.endswith('S'):
                                    resample_rule = f"{aggregation.rstrip('S')}S"
                                elif aggregation.endswith('T'):
                                    resample_rule = f"{aggregation.rstrip('T')}T"
                                else:
                                    resample_rule = '1S'  # Default
                                aggregated_df = df_viz.resample(resample_rule).mean().reset_index()
                            st.sidebar.write(f"Data aggregated every {aggregation}")
                        else:
                            st.sidebar.warning("Unknown aggregation interval. Skipping aggregation.")
                            aggregated_df = df_viz

                        create_features_vs_time(aggregated_df, selected_features, time_column, sampling_rate)
                    else:
                        st.warning("Please select features to visualize over time.")
                elif selected_option == 'Pressure Distribution' and time_column:
                    if 'Working pressure [bar]' in df_viz.columns:
                        renamed_pressure_col = 'Working pressure [bar]'
                        create_pressure_distribution_polar_plot(df_viz, renamed_pressure_col, time_column)
                    else:
                        st.warning("Please select a valid working pressure column.")
                elif selected_option == 'Parameters vs Chainage':
                    if selected_features:
                        # **Enhancement 1: Chainage Filtering & Data Averaging Based on Sampling Rate**
                        st.sidebar.header("Chainage Filtering & Data Averaging")

                        # User selects aggregation interval for chainage
                        aggregation = st.sidebar.selectbox(
                            "Select aggregation interval for Chainage",
                            ['None', '1S', '5S', '10S', '30S', '1T', '5T', '10T', '30T']
                        )

                        if aggregation != 'None':
                            # Chainage Filtering and Averaging
                            df_viz_processed = handle_chainage_filtering_and_averaging(df_viz, 'Chainage [mm]', aggregation)
                        else:
                            df_viz_processed = df_viz

                        create_parameters_vs_chainage(
                            df_viz_processed, 
                            selected_features, 
                            'Chainage_mid',  # Use 'Chainage_mid' for plotting
                            penetration_rates_available=('Penetration Rate [mm/rev]' in df_viz_processed.columns), 
                            aggregation=aggregation
                        )
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
    st.markdown("**Created by Kursat Kilic - Geotechnical Digitalization**")

def suggest_column(df, keywords):
    """Helper function to suggest columns based on keywords"""
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                return col
    return None

if __name__ == "__main__":
    main()
