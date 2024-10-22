import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
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

# Function to read and preprocess the data
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

# Function to identify special columns based on keywords
def identify_special_columns(df):
    working_pressure_keywords = ['working pressure', 'arbeitsdruck', 'sr_arbdr', 'SR_Arbdr','pressure', 'druck', 'arbdr']
    revolution_keywords = ['revolution', 'revolution (rpm)','drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz','Revolution', 'Revolution [rpm]','Revolution (rpm)']
    advance_rate_keywords = ['advance rate', 'advance rate [mm/min]', 'Advance rate', 'Advance_rate','Advance Rate','vortrieb', 'vorschub', 'VTgeschw_Z', 'geschw', 'geschw_Z']
    penetration_rate_keywords = ['penetration rate', 'pen rate', 'penetration', 'penetration_rate']

    working_pressure_cols = [col for col in df.columns if any(kw in col.lower() for kw in working_pressure_keywords)]
    revolution_cols = [col for col in df.columns if any(kw in col.lower() for kw in revolution_keywords)]
    advance_rate_cols = [col for col in df.columns if any(kw in col.lower() for kw in advance_rate_keywords)]
    penetration_rate_cols = [col for col in df.columns if any(kw in col.lower() for kw in penetration_rate_keywords)]

    return working_pressure_cols, revolution_cols, advance_rate_cols, penetration_rate_cols

# Helper function to suggest a column based on keywords
def suggest_column(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                return col
    return None

# Function to get distance/chainage columns
def get_distance_columns(df):
    distance_keywords = ['distance', 'length', 'travel', 'chainage', 'tunnellänge neu', 'tunnellänge', 'weg_mm_z', 'vtp_weg']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in distance_keywords)]

# Function to get time column
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
            df = df.sort_values(by=time_column)
            df[distance_column] = pd.to_numeric(df[distance_column], errors='coerce')
            df['Time_Diff'] = df[time_column].diff().dt.total_seconds()
            df['Distance_Diff'] = df[distance_column].diff()
            df['Average Speed (mm/min)'] = (df['Distance_Diff'] / df['Time_Diff']) * 60  # Convert to mm/min
            
            if revolution_col is not None:
                df['Calculated Penetration Rate [mm/rev]'] = df['Average Speed (mm/min)'] / df[revolution_col]
        
        return df
            
    except Exception as e:
        st.error(f"Error calculating derived features: {str(e)}")
        return df

# Function to add sensor-based penetration rate
def add_sensor_based_penetration(df, sensor_penetration_col):
    if sensor_penetration_col in df.columns:
        df['Sensor-based Penetration Rate [mm/rev]'] = df[sensor_penetration_col]
    else:
        st.warning("Sensor-based penetration rate column not found.")
    return df

# Function to rename columns for consistency
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

# Function to safely select a column with a suggested option
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
def handle_chainage_filtering_and_averaging(df, chainage_column):
    try:
        st.sidebar.header("Chainage Filtering & Averaging")

        # User input for chainage range
        min_chainage = float(df[chainage_column].min())
        max_chainage = float(df[chainage_column].max())
        chainage_range = st.sidebar.slider("Select Chainage Range (mm)", min_chainage, max_chainage, (min_chainage, max_chainage))

        # Filter the DataFrame based on selected range
        filtered_df = df[(df[chainage_column] >= chainage_range[0]) & (df[chainage_column] <= chainage_range[1])]

        st.sidebar.write(f"Selected Chainage Range: {chainage_range[0]} - {chainage_range[1]} mm")

        # User input for averaging interval
        averaging_units = {'Millimeters (mm)': 1, 'Centimeters (cm)': 10, 'Meters (m)': 1000}
        selected_unit = st.sidebar.selectbox("Select Averaging Unit", list(averaging_units.keys()))
        averaging_interval = st.sidebar.number_input(f"Averaging Interval in {selected_unit}", min_value=1, value=10)
        interval_in_mm = averaging_interval * averaging_units[selected_unit]

        # Aggregate data based on averaging interval
        bins = np.arange(filtered_df[chainage_column].min(), filtered_df[chainage_column].max() + interval_in_mm, interval_in_mm)
        filtered_df['Chainage Bin'] = pd.cut(filtered_df[chainage_column], bins=bins, include_lowest=True)
        aggregated_df = filtered_df.groupby('Chainage Bin').mean().reset_index()
        aggregated_df[chainage_column] = aggregated_df['Chainage Bin'].apply(lambda x: x.mid)

        return aggregated_df
    except Exception as e:
        st.error(f"Error in chainage filtering and averaging: {e}")
        return df

# Function to handle time filtering and averaging
def handle_time_filtering_and_averaging(df, time_column):
    try:
        st.sidebar.header("Time Filtering & Averaging")

        # User input for time range
        min_time = df[time_column].min()
        max_time = df[time_column].max()
        time_range = st.sidebar.slider("Select Time Range", min_value=min_time, max_value=max_time, value=(min_time, max_time), format="YYYY-MM-DD HH:mm:ss")

        # Filter the DataFrame based on selected time range
        filtered_df = df[(df[time_column] >= time_range[0]) & (df[time_column] <= time_range[1])]

        st.sidebar.write(f"Selected Time Range: {time_range[0]} - {time_range[1]}")

        # User input for averaging interval
        averaging_units = {'Seconds': 'S', 'Minutes': 'T', 'Hours': 'H'}
        selected_unit = st.sidebar.selectbox("Select Averaging Unit", list(averaging_units.keys()))
        averaging_interval = st.sidebar.number_input(f"Averaging Interval in {selected_unit}", min_value=1, value=1)
        interval = f"{averaging_interval}{averaging_units[selected_unit]}"

        # Aggregate data based on averaging interval
        filtered_df = filtered_df.set_index(time_column)
        aggregated_df = filtered_df.resample(interval).mean().reset_index()

        return aggregated_df
    except Exception as e:
        st.error(f"Error in time filtering and averaging: {e}")
        return df

# Function to create Parameters vs Chainage plot
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
              '#9B6B6B', '#E9967A', '#4682B4', '#6B8E23']

    available_features = [f for f in selected_features if f in df.columns]
    
    if not available_features:
        st.warning("None of the selected features are available in the dataset.")
        return

    fig = make_subplots(rows=len(available_features), cols=1,
                        shared_xaxes=True,
                        subplot_titles=available_features,
                        vertical_spacing=0.05)

    for i, feature in enumerate(available_features, start=1):
        try:
            y_data = df[feature]
            feature_name = feature

            fig.add_trace(
                go.Scatter(
                    x=df[chainage_column],
                    y=y_data,
                    mode='lines+markers',
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

    # Update layout
    fig.update_layout(
        height=400 * len(available_features),
        width=1200,
        title_text=f'Parameters vs Chainage',
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    # Update x-axis title
    fig.update_xaxes(title_text='Chainage [mm]', row=len(available_features), col=1)

    st.plotly_chart(fig, use_container_width=True)

# Function to create Thrust Force plots
def create_thrust_force_plots(df, advance_rate_col, num_cutting_rings):
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

        # Divide thrust force by number of cutting rings
        df['Thrust Force per Ring'] = df[thrust_force_col] / num_cutting_rings

        # Create subplots
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=("Thrust Force vs Calculated Penetration Rate", 
                                         "Thrust Force vs Average Speed", 
                                         "Thrust Force vs Advance Rate"),
                           vertical_spacing=0.1)

        # Plot 1: Thrust Force vs Calculated Penetration Rate
        if 'Calculated Penetration Rate [mm/rev]' in df.columns:
            mask = df['Calculated Penetration Rate [mm/rev]'].notna()
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'Calculated Penetration Rate [mm/rev]'], 
                y=df.loc[mask, 'Thrust Force per Ring'], 
                mode='markers', 
                name='vs Calculated Penetration Rate', 
                marker=dict(color='blue', size=5)
            ), row=1, col=1)
        else:
            st.warning("Calculated Penetration Rate [mm/rev] column not found in the dataset.")

        # Plot 2: Thrust Force vs Average Speed
        if 'Average Speed (mm/min)' in df.columns:
            mask = df['Average Speed (mm/min)'].notna()
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'Average Speed (mm/min)'], 
                y=df.loc[mask, 'Thrust Force per Ring'], 
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
                y=df.loc[mask, 'Thrust Force per Ring'], 
                mode='markers', 
                name='vs Advance Rate',
                marker=dict(color='red', size=5)
            ), row=3, col=1)
        else:
            st.warning("Selected advance rate column not available for plotting.")

        # Update layout
        fig.update_layout(
            height=1200, 
            width=800, 
            title_text="Thrust Force Relationships",
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Calculated Penetration Rate [mm/rev]", row=1, col=1)
        fig.update_xaxes(title_text="Average Speed (mm/min)", row=2, col=1)
        fig.update_xaxes(title_text=advance_rate_col if advance_rate_col else "Advance Rate [mm/min]", row=3, col=1)
        
        for i in range(1, 4):
            fig.update_yaxes(title_text="Thrust Force per Ring [kN]", row=i, col=1)

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating thrust force plots: {e}")

# Function to create Features vs Time plot
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
                mode='lines+markers',
                name=feature,
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=i,
            col=1
        )

        # Update y-axis titles
        fig.update_yaxes(title_text=feature, row=i, col=1)

    # Update x-axis titles dynamically
    fig.update_xaxes(title_text='Time', row=len(selected_features), col=1)

    # Update layout with larger dimensions and better spacing
    fig.update_layout(
        height=400 * len(selected_features),  # Increased height per subplot
        width=1200,  # Increased overall width
        title_text='Features vs Time',
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to create statistical summary
def create_statistical_summary(df, selected_features):
    if not selected_features:
        st.warning("Please select at least one feature for the statistical summary.")
        return

    summary = df[selected_features].describe().transpose()

    st.write(summary)

# Function to create polar plot
def create_pressure_distribution_polar_plot(df, pressure_column, time_column):
    try:
        # Check if the pressure column exists
        if pressure_column not in df.columns:
            st.warning(f"Pressure column '{pressure_column}' not found in the dataset.")
            return

        df[pressure_column] = pd.to_numeric(df[pressure_column], errors='coerce')

        # Normalize time to angles
        df['Time_Angle'] = (df[time_column] - df[time_column].min()) / (df[time_column].max() - df[time_column].min()) * 360

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=df[pressure_column],
            theta=df['Time_Angle'],
            mode='markers',
            marker=dict(color='blue', size=5),
            name='Pressure'
        ))

        fig.update_layout(
            title='Pressure Distribution Over Time (Polar Plot)',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[df[pressure_column].min(), df[pressure_column].max()]
                )
            ),
            showlegend=False
        )

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating polar plot: {e}")

# Function to read and process rock strength data
def read_rock_strength_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error reading rock strength data: {e}")
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

# Function to create rock strength comparison chart
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

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating rock strength comparison chart: {e}")

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
                working_pressure_cols, revolution_cols, advance_rate_cols, penetration_rate_cols = identify_special_columns(df)

                suggested_working_pressure = suggest_column(df, ['working pressure', 'arbeitsdruck', 'pressure', 'druck', 'arbdr', 'sr_arbdr','SR_Arbdr'])
                suggested_revolution = suggest_column(df, ['revolution', 'drehzahl', 'rpm', 'drehz', 'sr_drehz', 'SR_Drehz'])
                suggested_advance_rate = suggest_column(df, ['advance rate', 'vortrieb', 'vorschub','VTgeschw','geschw'])
                suggested_penetration_rate = suggest_column(df, ['penetration rate', 'pen rate', 'penetration', 'penetration_rate'])

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

                # Input for number of cutting rings
                num_cutting_rings = st.sidebar.number_input("Enter number of cutting rings", min_value=1, value=1, step=1)

                if working_pressure_col != 'None' and revolution_col != 'None':
                    df = calculate_derived_features(df, working_pressure_col, revolution_col, n1, torque_constant, selected_distance)
                    
                # Add sensor-based penetration rate if available
                if suggested_penetration_rate is not None:
                    df = add_sensor_based_penetration(df, suggested_penetration_rate)

                df_viz = rename_columns(df.copy(), working_pressure_col, revolution_col, selected_distance, advance_rate_col)

                all_features = df_viz.columns.tolist()
                
                time_column = get_time_column(df_viz)

                options = ['Parameters vs Chainage', 'Features vs Time', 'Thrust Force Plots', 'Statistical Summary', 'Pressure Distribution', 'Rock Strength Comparison']
                selected_option = st.sidebar.radio("Choose visualization", options)

                default_features = []
                if 'Calculated torque [kNm]' in all_features:
                    default_features.append('Calculated torque [kNm]')
                if 'Average Speed (mm/min)' in all_features:
                    default_features.append('Average Speed (mm/min)')
                if 'Calculated Penetration Rate [mm/rev]' in all_features:
                    default_features.append('Calculated Penetration Rate [mm/rev]')
                if 'Sensor-based Penetration Rate [mm/rev]' in all_features:
                    default_features.append('Sensor-based Penetration Rate [mm/rev]')
                
                selected_features = st.sidebar.multiselect(
                    "Select features for analysis",
                    all_features,
                    default=default_features
                )

                st.subheader(f"Visualization: {selected_option}")

                if selected_option == 'Thrust Force Plots':
                    create_thrust_force_plots(
                        df_viz, 
                        'Advance rate [mm/min]' if advance_rate_col != 'None' else None,
                        num_cutting_rings
                    )
                elif selected_option == 'Features vs Time':
                    if selected_features:
                        df_viz = handle_time_filtering_and_averaging(df_viz, time_column)
                        create_features_vs_time(df_viz, selected_features, time_column)
                    else:
                        st.warning("Please select features to visualize over time.")
                elif selected_option == 'Parameters vs Chainage':
                    if selected_features:
                        df_viz = handle_chainage_filtering_and_averaging(df_viz, 'Chainage [mm]')

                        create_parameters_vs_chainage(df_viz, selected_features, 'Chainage [mm]')
                    else:
                        st.warning("Please select features to visualize against chainage.")
                elif selected_option == 'Statistical Summary':
                    if selected_features:
                        create_statistical_summary(df_viz, selected_features)
                    else:
                        st.warning("Please select features for statistical summary.")
                elif selected_option == 'Pressure Distribution':
                    if working_pressure_col != 'None' and time_column is not None:
                        create_pressure_distribution_polar_plot(df_viz, 'Working pressure [bar]', time_column)
                    else:
                        st.warning("Please ensure both working pressure and time columns are selected.")
                elif selected_option == 'Rock Strength Comparison':
                    if rock_strength_file is not None:
                        rock_strength_data = read_rock_strength_data(rock_strength_file)
                        if rock_strength_data is not None:
                            rock_df = preprocess_rock_strength_data(rock_strength_data)
                            if rock_df is not None:
                                rock_type = st.sidebar.selectbox("Select Rock Type", rock_df.index)
                                create_rock_strength_comparison_chart(df_viz, rock_df, rock_type, selected_features)
                            else:
                                st.warning("Error processing rock strength data.")
                        else:
                            st.warning("Error loading rock strength data.")
                    else:
                        st.warning("Please upload rock strength data to use this feature.")

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

if __name__ == "__main__":
    main()

