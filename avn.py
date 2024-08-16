import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.interpolate import griddata
from scipy import stats

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
    
    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, subplot_titles=features)
    for i, feature in enumerate(features, start=1):
        fig.add_trace(go.Scatter(x=df['Relative time'], y=df[feature], mode='lines', name=feature), row=i, col=1)

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
    
    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, subplot_titles=features)
    for i, feature in enumerate(features, start=1):
        fig.add_trace(go.Scatter(x=df['Chainage'], y=df[feature], mode='lines', name=feature), row=i, col=1)

    fig.update_layout(height=300 * len(features), width=1000, title_text='Parameters vs Chainage')
    st.plotly_chart(fig)

# Function to create 3D spectrogram with proper 3D spectrum visualization
def create_3d_spectrogram(df):
    x = df['Working pressure [bar]'].values
    y = df['Revolution [rpm]'].values
    z = df['Calculated torque [kNm]'].values

    # Creating a grid for smoother interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

    # 3D surface plot
    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale='Viridis')])

    fig.update_layout(
        title='3D Spectrogram of Key Parameters',
        scene=dict(
            xaxis_title='Working Pressure [bar]',
            yaxis_title='Revolution [rpm]',
            zaxis_title='Calculated Torque [kNm]',
            xaxis=dict(nticks=10),
            yaxis=dict(nticks=10),
            zaxis=dict(nticks=10)
        ),
        autosize=False,
        width=800,
        height=800,
    )

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
    colors = ['blue', 'red', 'green', 'orange']

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
    colors = ['blue', 'red', 'green', 'orange']

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

# Streamlit app
def main():
    st.title("Enhanced Machine Parameter Analysis and Rock Strength Comparison")

    # Sidebar for file upload
    st.sidebar.header("Upload your data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

    rock_strength_file = st.sidebar.file_uploader("Upload Rock Strength Excel File", type=['xlsx'])
    rock_df = None
    rock_strength_data = None
    if rock_strength_file:
        rock_strength_data = read_rock_strength_data(rock_strength_file)
        if rock_strength_data is not None:
            rock_df = preprocess_rock_strength_data(rock_strength_data)
            st.sidebar.subheader("Select Rock Type")
            rock_type = st.sidebar.selectbox("Rock Type", rock_df.index)

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            # Sidebar navigation for visualization
            st.sidebar.header("Select Visualization")
            options = st.sidebar.radio("Choose visualization type", [
                'Correlation Heatmap', 'Statistical Summary', '3D Spectrogram', 
                'Features vs Time', 'Pressure Distribution Polar Plot', 'Parameters vs Chainage', 
                'Box Plots', 'Violin Plots', 'Rock Strength Comparison'
            ])

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

                numeric_columns = ['Working pressure [bar]', 'Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Relative time', 'Weg VTP [mm]', 'SR Position [Grad]']
                for col in numeric_columns:
                    df = clean_numeric_column(df, col)

            # Calculate Penetration_Rate and Calculated torque
            df['Penetration_Rate'] = df['Advance rate (mm/min)'] / df['Revolution [rpm]']
            df['Calculated torque [kNm]'] = df['Working pressure [bar]'] * 0.1  # Assuming a linear relationship

            # Display selected visualization
            if options == 'Correlation Heatmap':
                create_correlation_heatmap(df)
            elif options == 'Statistical Summary':
                create_statistical_summary(df)
            elif options == '3D Spectrogram':
                create_3d_spectrogram(df)
            elif options == 'Features vs Time':
                create_features_vs_time(df)
            elif options == 'Pressure Distribution Polar Plot':
                create_polar_plot(df)
            elif options == 'Parameters vs Chainage':
                create_parameters_vs_chainage(df)
            elif options == 'Box Plots':
                create_multi_axis_box_plots(df)
            elif options == 'Violin Plots':
                create_multi_axis_violin_plots(df)
            elif options == 'Rock Strength Comparison' and rock_df is not None:
                create_rock_strength_comparison_chart(df, rock_df, rock_type)
    else:
        st.info("Please upload a CSV or Excel file to begin.")

if __name__ == "__main__":
    main()
