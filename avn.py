import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

st.set_option('deprecation.showPyplotGlobalUse', False)

# Helper function to clean numeric columns
def clean_numeric_column(df, column_name):
    df[column_name] = df[column_name].replace(r'[^0-9.-]+', '', regex=True)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

# Function to read CSV or Excel file
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        return None

# Function to preprocess the rock strength data
def preprocess_rock_strength_data(df):
    df['Rock Type'] = df['Probenbezeichnung'].str.split().str[0]
    pivoted = df.pivot_table(values='Value', index='Rock Type', columns='Test', aggfunc='mean')
    pivoted.rename(columns={'UCS': 'UCS (MPa)', 'BTS': 'BTS (MPa)', 'PLT': 'PLT (MPa)'}, inplace=True)
    return pivoted

# Function to visualize correlation heatmap
def create_correlation_heatmap(df):
    features = ['Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Calculated torque [kNm]', 'Penetration_Rate', 'Working pressure [bar]']
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Selected Parameters')
    plt.tight_layout()
    st.pyplot()

# Function to create statistical summary
def create_statistical_summary(df, round_to=2):
    features = ['Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]']
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
            'max': round(df[feature].max(), round_to)
        }

    summary = pd.DataFrame(summary_dict).transpose()
    st.dataframe(summary)

# Function to create 3D spectrogram
def create_3d_spectrogram(df):
    x = df['Working pressure [bar]'].values
    y = df['Revolution [rpm]'].values
    z = df['Calculated torque [kNm]'].values

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale='Viridis')])

    fig.update_layout(
        title='3D Spectrogram of Key Parameters',
        scene=dict(
            xaxis_title='Working Pressure [bar]',
            yaxis_title='Revolution [rpm]',
            zaxis_title='Calculated Torque [kNm]'
        ),
        autosize=False,
        width=800,
        height=800,
    )

    st.plotly_chart(fig)

# Function to create multi-axis box plots
def create_multi_axis_box_plots(df):
    features = ['Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]']
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    colors = ['blue', 'red', 'green', 'orange']

    for i, feature in enumerate(features):
        if i < 2:
            fig.add_trace(go.Box(y=df[feature], name=feature, marker_color=colors[i]), secondary_y=False)
        else:
            fig.add_trace(go.Box(y=df[feature], name=feature, marker_color=colors[i]), secondary_y=True)

    fig.update_layout(
        title='Box Plots of Key Parameters',
        height=600,
        width=1000,
        showlegend=True,
        boxmode='group'
    )
    st.plotly_chart(fig)

# Function to create multi-axis violin plots
def create_multi_axis_violin_plots(df):
    features = ['Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]']
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    colors = ['blue', 'red', 'green', 'orange']

    for i, feature in enumerate(features):
        if i < 2:
            fig.add_trace(go.Violin(y=df[feature], name=feature, box_visible=True, meanline_visible=True, fillcolor=colors[i]), secondary_y=False)
        else:
            fig.add_trace(go.Violin(y=df[feature], name=feature, box_visible=True, meanline_visible=True, fillcolor=colors[i]), secondary_y=True)

    fig.update_layout(
        title='Violin Plots of Key Parameters',
        height=600,
        width=1000,
        showlegend=True,
        violinmode='group'
    )
    st.plotly_chart(fig)

# Streamlit app
st.title("Machine Parameter Analysis and Rock Strength Comparison")

# Sidebar for file upload
st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        # Sidebar navigation for visualization
        st.sidebar.header("Select Visualization")
        options = st.sidebar.radio("Choose visualization type", ['Correlation Heatmap', 'Statistical Summary', '3D Spectrogram', 'Box Plots', 'Violin Plots'])

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

        # Sidebar for rock strength data if Excel is uploaded
        if uploaded_file.name.endswith('.xlsx'):
            rock_strength_df = preprocess_rock_strength_data(df)
            st.sidebar.subheader("Select Rock Type")
            rock_type = st.sidebar.selectbox("Rock Type", rock_strength_df.index)

        # Display selected visualization
        if options == 'Correlation Heatmap':
            create_correlation_heatmap(df)
        elif options == 'Statistical Summary':
            create_statistical_summary(df)
        elif options == '3D Spectrogram':
            create_3d_spectrogram(df)
        elif options == 'Box Plots':
            create_multi_axis_box_plots(df)
        elif options == 'Violin Plots':
            create_multi_axis_violin_plots(df)
else:
    st.info("Please upload a CSV or Excel file to begin.")
