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
    page_icon="https://github.com/kilickursat/AVNanalyzer/blob/main/Herrenknecht_logo.svg-1024x695.png?raw=true"
)

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

# Updated function to create comparison chart for machine parameters vs rock strength
def create_rock_strength_comparison_chart(machine_df, rock_df, rock_type, selected_features):
    rock_df = rock_df[rock_df.index == rock_type]
    if rock_df.empty:
        st.error(f"No data available for {rock_type} rock type.")
        return

    avg_values = machine_df[selected_features].mean()

    parameters = selected_features
    machine_values = avg_values.values
    ucs_values = [rock_df['UCS (MPa)'].iloc[0]] * len(selected_features)
    bts_values = [rock_df['BTS (MPa)'].iloc[0]] * len(selected_features)
    plt_values = [rock_df['PLT (MPa)'].iloc[0]] * len(selected_features)

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
    
    colors = {
        'Penetration_Rate': '#98fb98',
        'Calculated torque [kNm]': '#4b0082',
        'Working pressure [bar]': '#ff00ff',
        'Revolution [rpm]': '#0000cd',
        'Thrust force [kN]': '#6495ed'
    }

    fig = make_subplots(rows=len(selected_features), cols=1, shared_xaxes=True, subplot_titles=selected_features)
    for i, feature in enumerate(selected_features, start=1):
        fig.add_trace(go.Scatter(x=df[time_column], y=df[feature], mode='lines', name=feature, line=dict(color=colors.get(feature, '#000000'))), row=i, col=1)

    fig.update_layout(height=300 * len(selected_features), width=1000, title_text='Features vs Time')
    st.plotly_chart(fig)

# Updated function to create Pressure Distribution Over Time Polar Plot with Plotly
def create_pressure_distribution_polar_plot(df, pressure_column, time_column):
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

# Updated function to create Parameters vs Chainage plot with Plotly subplots
def create_parameters_vs_chainage(df, selected_features):
    if not selected_features:
        st.warning("Please select at least one feature for the chainage plot.")
        return

    df = df.sort_values(by='Chainage')  # Sort the data by Chainage

    colors = {
        'Penetration_Rate': '#98fb98', # palegreen
        'Calculated torque [kNm]': '#4b0082', # indigo
        'Revolution [rpm]': '#0000cd', # mediumblue
        'Thrust force [kN]': '#6495ed' # cornflowerblue
    }
    
    fig = make_subplots(rows=len(selected_features), cols=1, shared_xaxes=True, subplot_titles=selected_features)
    for i, feature in enumerate(selected_features, start=1):
        fig.add_trace(go.Scatter(x=df['Chainage'], y=df[feature], mode='lines', name=feature, line=dict(color=colors.get(feature, '#000000'))), row=i, col=1)

    fig.update_layout(height=300 * len(selected_features), width=1000, title_text='Parameters vs Chainage')
    st.plotly_chart(fig)

# Updated function to create multi-axis box plots with additional features
def create_multi_axis_box_plots(df, selected_features):
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

# Updated function to create multi-axis violin plots with added customization
def create_multi_axis_violin_plots(df, selected_features):
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


def get_distance_columns(df):
    distance_keywords = ['distance', 'length', 'travel', 'chainage']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in distance_keywords)]


def create_thrust_force_plots(df):
    thrust_force_col = suggest_column(df, ['thrust force', 'vorschubkraft', 'kraft','Kraft','Kraft_max','GesamtKraft','GesamKraft_STZ','GesamKraft_VTP'])
    penetration_rate_col = suggest_column(df, ['penetration rate', 'penetrationsrate', 'penetration rate [mm/rev]', 'Penetration_Rate [mm/rev]', 'Penetration_Rate[mm/rev]'])
    advance_rate_col = suggest_column(df, ['advance rate', 'vortriebsgeschwindigkeit', 'vortrieb','VTP_Weg','Weg'])

    if not thrust_force_col:
        st.warning("Thrust force column not found in the dataset.")
        return

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Thrust Force vs Penetration Rate", "Thrust Force vs Advance Rate"))

    if penetration_rate_col:
        fig.add_trace(go.Scatter(x=df[thrust_force_col], y=df[penetration_rate_col], mode='markers', name='Penetration Rate'), row=1, col=1)
    else:
        st.warning("Penetration rate column not found in the dataset.")

    if advance_rate_col:
        fig.add_trace(go.Scatter(x=df[thrust_force_col], y=df[advance_rate_col], mode='markers', name='Advance Rate'), row=2, col=1)
    else:
        st.warning("Advance rate column not found in the dataset.")

    fig.update_layout(height=800, width=800, title_text="Thrust Force Relationships")
    fig.update_xaxes(title_text="Thrust Force", row=1, col=1)
    fig.update_xaxes(title_text="Thrust Force", row=2, col=1)
    fig.update_yaxes(title_text="Penetration Rate", row=1, col=1)
    fig.update_yaxes(title_text="Advance Rate", row=2, col=1)

    st.plotly_chart(fig)

def calculate_derived_features(df, working_pressure_col, advance_rate_col, revolution_col, n1, torque_constant):
    if working_pressure_col and revolution_col:
        def calculate_torque_wrapper(row):
            working_pressure = row[working_pressure_col]
            current_speed = row[revolution_col]
            if current_speed < n1:
                torque = working_pressure * torque_constant
            else:
                torque = (n1 / current_speed) * torque_constant * working_pressure
            return round(torque, 2)
        
        df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)

    # Calculate penetration rate if advance rate and revolution are available
    if advance_rate_col and revolution_col:
        df['Penetration_Rate [mm/rev]'] = df[advance_rate_col] / df[revolution_col]

    return df


# Streamlit app
def main():
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
            suggested_advance_rate = suggest_column(df, ['advance rate', 'vortrieb', 'vorschub','VTgeschw','geschw'])

            # Let user select working pressure, revolution, and advance rate columns
            working_pressure_col = st.sidebar.selectbox(
                "Select Working Pressure Column", 
                ['None'] + working_pressure_cols,
                index=working_pressure_cols.index(suggested_working_pressure) + 1 if suggested_working_pressure else 0
            )
            revolution_col = st.sidebar.selectbox(
                "Select Revolution Column", 
                ['None'] + revolution_cols,
                index=revolution_cols.index(suggested_revolution) + 1 if suggested_revolution else 0
            )
            advance_rate_col = st.sidebar.selectbox(
                "Select Advance Rate Column", 
                ['None'] + advance_rate_cols,
                index=advance_rate_cols.index(suggested_advance_rate) + 1 if suggested_advance_rate else 0
            )

            # Add input fields for n1 and torque_constant
            n1 = st.sidebar.number_input("Enter n1 value (revolution 1/min)", min_value=0.0, value=1.0, step=0.1)
            torque_constant = st.sidebar.number_input("Enter torque constant", min_value=0.0, value=1.0, step=0.1)

            # Calculate derived features if possible
            if working_pressure_col != 'None' or (advance_rate_col != 'None' and revolution_col != 'None'):
                df = calculate_derived_features(df, 
                                                working_pressure_col if working_pressure_col != 'None' else None,
                                                advance_rate_col if advance_rate_col != 'None' else None,
                                                revolution_col if revolution_col != 'None' else None,
                                                n1,
                                                torque_constant)

            # Let user select features for analysis
            selected_features = st.sidebar.multiselect("Select features for analysis", df.columns)

            # Check for time-related columns
            time_column = get_time_column(df)

            # Visualization selection
            options = ['Correlation Heatmap', 'Statistical Summary', 'Parameters vs Chainage', 'Box Plots', 'Violin Plots']
            
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

            # Visualization based on user selection
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
                distance_columns = get_distance_columns(df)
                if distance_columns:
                    selected_distance = st.selectbox("Select distance feature", distance_columns)
                    create_parameters_vs_chainage(df, selected_features, selected_distance)
                else:
                    st.warning("No distance-related columns found in the dataset.")
            elif selected_option == 'Box Plots':
                create_multi_axis_box_plots(df, selected_features)
            elif selected_option == 'Violin Plots':
                create_multi_axis_violin_plots(df, selected_features)
            elif selected_option == 'Rock Strength Comparison':
                if rock_df is not None and 'rock_type' in locals():
                    create_rock_strength_comparison_chart(df, rock_df, rock_type, selected_features)
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
    st.markdown("© 2024 Herrenknecht AG. All rights reserved.")
    st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

if __name__ == "__main__":
    main()
