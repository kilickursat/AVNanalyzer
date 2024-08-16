import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata


# Function to read the CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'], keep_default_na=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function to calculate torque
def calculate_torque(working_pressure, torque_constant, current_speed=None, n1=None):
    if current_speed is None or n1 is None:
        # Without Variable Speed Motor (n2 = 0)
        torque = working_pressure * torque_constant
    else:
        if current_speed < n1:
            # With Variable Speed Motor, Current Speed < n1
            torque = working_pressure * torque_constant
        else:
            # With Variable Speed Motor, Current Speed > n1
            torque = (n1 / current_speed) * torque_constant * working_pressure

    return torque


def calculate_torque_wrapper(row):
    working_pressure = row['Working pressure [bar]']
    current_speed = row['Revolution [rpm]']
    torque_constant = 0.14376997
    n1 = 25.7

    return round(calculate_torque(working_pressure, torque_constant, current_speed, n1), 2)


def calculate_penetration_rate(row):
    speed = row['Average Speed (mm/min)']
    revolution = row['Revolution [rpm]']

    if pd.isna(speed) or pd.isna(revolution):
        return np.nan
    elif revolution == 0:
        return np.inf if speed != 0 else 0
    else:
        return round(speed / revolution, 4)


def calculate_advance_rate_and_stats(df):
    distance_column = 'Weg VTP [mm]'
    time_column = 'Relative time'

    if len(df) > 1:
        weg = round(df[distance_column].max() - df[distance_column].min(), 2)
        zeit = round(df[time_column].max() - df[time_column].min(), 2)
    else:
        weg = df[distance_column].iloc[0]
        zeit = df[time_column].iloc[0]

    zeit = zeit * (0.000001 / 60)

    average_speed = round(weg / zeit, 2) if zeit != 0 else 0

    print(f"Total Distance (weg): {weg} mm")
    print(f"Total Time (zeit): {zeit} minutes")
    print(f"Average Speed: {average_speed} mm/min")

    result = {
        "Total Distance (mm)": weg,
        "Total Time (min)": zeit,
        "Average Speed (mm/min)": average_speed
    }

    return result, average_speed


# Function to create a correlation heatmap
def create_correlation_heatmap(df):
    features = ['Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Calculated torque [kNm]', 'Penetration_Rate', 'Working pressure [bar]']
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Selected Parameters')
    plt.tight_layout()
    plt.show()


def perform_ols_regression(df, x_col, y_col):
    X = sm.add_constant(df[x_col])
    model = sm.OLS(df[y_col], X).fit()
    print(f"\nOLS Regression Results ({y_col} vs {x_col}):")
    print(model.summary())

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.plot(df[x_col], model.predict(X), color='red', linewidth=2)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} vs {x_col}')
    equation = f'y = {model.params[1]:.4f}x + {model.params[0]:.4f}'
    plt.text(0.05, 0.95, f'R² = {model.rsquared:.4f}\n{equation}',
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.tight_layout()
    plt.show()

    return model


# Function to create a polar plot
def create_polar_plot(df):
    pressure_column = 'Revolution [rpm]'
    time_normalized = np.linspace(0, 360, len(df))
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
        title='Pressure Distribution Over Time',
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
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                direction='clockwise',
                rotation=90,
                gridcolor='lightgrey'
            )
        ),
        showlegend=False,
        template='plotly_white'
    )
    fig.show()


# Function to preprocess data
def preprocess_data(df, chainage_col, param_col):
    df_grouped = df.groupby(chainage_col).last().reset_index()
    chainage_min = df_grouped[chainage_col].min()
    chainage_max = df_grouped[chainage_col].max()
    chainage_interp = np.linspace(chainage_min, chainage_max, num=10000)
    param_interp = np.interp(chainage_interp, df_grouped[chainage_col], df_grouped[param_col])
    return chainage_interp, param_interp


# Function to visualize parameters vs chainage
def visualize_parameters_vs_chainage(df):
    features = ['Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Average Speed (mm/min)', 'Thrust force [kN]']
    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=features)

    for i, feature in enumerate(features, start=1):
        if feature == 'Average Speed (mm/min)':
            chainage_interp = df['Chainage']
            param_interp = [df['Average Speed (mm/min)'].iloc[0]] * len(df)
        else:
            chainage_interp, param_interp = preprocess_data(df, 'Chainage', feature)
        fig.add_trace(go.Scatter(x=chainage_interp, y=param_interp, mode='lines', name=feature), row=i, col=1)
        fig.update_yaxes(title_text=feature, row=i, col=1)

    fig.update_layout(height=1200, title_text="Parameters vs Chainage", showlegend=False)
    fig.update_xaxes(title_text="Chainage (m)", row=len(features), col=1)
    fig.show()


def create_statistical_summary(df, round_to=2):
    features = ['Revolution [rpm]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Thrust force [kN]']

    # Initialize an empty dictionary to store our results
    summary_dict = {}

    for feature in features:
        summary_dict[feature] = {
            'count': int(df[feature].count()),  # Count should be an integer
            'mean': round(df[feature].mean(), round_to),
            'median': round(df[feature].median(), round_to),
            'std': round(df[feature].std(), round_to),
            'min': round(df[feature].min(), round_to),
            '25%': round(df[feature].quantile(0.25), round_to),
            '50%': round(df[feature].quantile(0.50), round_to),
            '75%': round(df[feature].quantile(0.75), round_to),
            'max': round(df[feature].max(), round_to)
        }

    # Convert the dictionary to a DataFrame
    summary = pd.DataFrame(summary_dict)

    # Transpose the DataFrame so features are columns
    summary = summary.transpose()

    # Reorder the columns
    column_order = ['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
    summary = summary[column_order]

    # Create the Plotly table with improved layout
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Statistic'] + list(summary.columns),
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)),
        cells=dict(values=[summary.index] + [summary[col] for col in summary.columns],
                   fill_color='lavender',
                   align='left',
                   font=dict(size=11)),
        columnwidth=[200] + [100] * len(summary.columns))
    ])

    fig.update_layout(
        title='Statistical Summary of Parameters',
        width=1200,  # Increase overall width
        height=400,  # Adjust height as needed
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins
    )

    fig.show()

    return summary  # Optionally return the summary DataFrame


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

    fig.show()


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

    fig.update_yaxes(title_text="Revolution [rpm] / Penetration_Rate", secondary_y=False)
    fig.update_yaxes(title_text="Calculated torque [kNm] / Thrust force [kN]", secondary_y=True)

    fig.show()


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

    fig.update_yaxes(title_text="Revolution [rpm] / Penetration_Rate", secondary_y=False)
    fig.update_yaxes(title_text="Calculated torque [kNm] / Thrust force [kN]", secondary_y=True)

    fig.show()


def create_density_heatmap(df):
    x = df['Working pressure [bar]']
    y = df['Penetration_Rate']

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=3, color=z, colorscale='Viridis', showscale=True)
    ))

    fig.update_layout(
        title='Density Heatmap: Working Pressure vs Penetration Rate',
        xaxis_title='Working Pressure',
        yaxis_title='Penetration Rate',
        coloraxis_colorbar=dict(title='Density')
    )
    fig.show()


def clean_numeric_column(df, column_name):
    df[column_name] = df[column_name].replace(r'[^0-9.-]+', '', regex=True)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df


def visualize_torque_components(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Relative time'], y=df['Working pressure [bar]'],
                             mode='lines', name='Working Pressure'))
    fig.add_trace(go.Scatter(x=df['Relative time'], y=df['Revolution [rpm]'],
                             mode='lines', name='Revolution'))
    fig.add_trace(go.Scatter(x=df['Relative time'], y=df['Calculated torque [kNm]'],
                             mode='lines', name='Calculated Torque'))

    fig.update_layout(title='Torque Components Over Time',
                      xaxis_title='Relative time',
                      yaxis_title='Value',
                      legend_title='Parameters')
    fig.show()


def create_torque_vs_pressure_plot(df):
    fig = px.scatter(df, x='Working pressure [bar]', y='Calculated torque [kNm]',
                     color='Revolution [rpm]',
                     labels={'Working pressure [bar]': 'Working pressure [bar]',
                             'Calculated torque [kNm]': 'Calculated torque [kNm]',
                             'Revolution [rpm]': 'Revolution [rpm]'},
                     title='Torque [kNm] vs Working Pressure [bar]')
    fig.show()


def read_rock_strength_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Rock strength data loaded successfully.")
        return df
    except Exception as e:
        print(f"An error occurred while reading rock strength data: {e}")
        return None


def preprocess_rock_strength_data(df):
    df['Rock Type'] = df['Probenbezeichnung'].str.split().str[0]
    pivoted = df.pivot_table(values='Value', index='Rock Type', columns='Test', aggfunc='mean')
    pivoted.rename(columns={'UCS': 'UCS (MPa)', 'BTS': 'BTS (MPa)', 'PLT': 'PLT (MPa)'}, inplace=True)
    return pivoted


def create_rock_strength_comparison_chart(machine_df, rock_df):
    granit_df = rock_df[rock_df.index == 'Sandstein']
    if granit_df.empty:
        print("No data for Sandstein rock type.")
        return

    avg_advance_rate = machine_df['Average Speed (mm/min)'].mean()
    avg_penetration_rate = machine_df['Penetration_Rate'].mean()
    avg_torque = machine_df['Calculated torque [kNm]'].mean()
    avg_pressure = machine_df['Working pressure [bar]'].mean()

    parameters = ['Advance Rate\n(mm/min)', 'Penetration Rate\n(mm/rev)', 'Torque\n(kNm)', 'Working Pressure\n(bar)']
    machine_values = [avg_advance_rate, avg_penetration_rate, avg_torque, avg_pressure]
    ucs_values = [granit_df['UCS (MPa)'].iloc[0]] * 4
    bts_values = [granit_df['BTS (MPa)'].iloc[0]] * 4
    plt_values = [granit_df['PLT (MPa)'].iloc[0]] * 4

    fig, axs = plt.subplots(2, 2, figsize=(16, 16), dpi=600)
    fig.suptitle("Machine Parameters vs Sandwich Rock Strength", fontsize=20, fontweight='bold')

    colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5']

    for i, (param, ax) in enumerate(zip(parameters, axs.flat)):
        x = np.arange(4)
        width = 0.2

        bars = ax.bar(x, [machine_values[i], ucs_values[i], bts_values[i], plt_values[i]], width, color=colors, edgecolor='black')

        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Sandwich - {param}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)

        machine_param_name = param.split('\n')[0]
        ax.set_xticklabels([machine_param_name, 'UCS', 'BTS', 'PLT'], fontsize=12, fontweight='bold')

        ax.tick_params(axis='y', labelsize=10)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('machine_parameters_vs_sandwich_rock_strength_updated.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'machine_parameters_vs_sandwich_rock_strength_updated.png'")


def visualize_data(df, average_speed):
    create_statistical_summary(df)
    create_3d_spectrogram(df)
    create_multi_axis_box_plots(df)
    create_multi_axis_violin_plots(df)


def main():
    file_path = "/content/drive/MyDrive/SW/BN SW 27 700.csv"
    df = read_csv_file(file_path)
    if df is None or df.empty:
        print("No data to process. Exiting.")
        return

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

    results, average_speed = calculate_advance_rate_and_stats(df)

    df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)
    df['Average Speed (mm/min)'] = average_speed
    df['Penetration_Rate'] = df.apply(calculate_penetration_rate, axis=1)

    visualize_data(df, average_speed)

    print("Summary Statistics:")
    print(results)
    print("\nGrouped Data (Mean Pressure by Degree):")

    try:
        print(df.groupby('SR Position [Grad]').mean()['Working pressure [bar]'])
    except Exception as e:
        print(f"Error calculating mean pressure by degree: {e}")
        print("Displaying first few rows of Arbeitsdruck instead:")
        print(df['Working pressure [bar]'].head())

    if len(df) > 1:
        visualize_machine_features(df)
        create_correlation_heatmap(df)
        model1 = perform_ols_regression(df, 'Penetration_Rate', 'Thrust force [kN]')
        model2 = perform_ols_regression(df, 'Advance rate (mm/min)', 'Thrust force [kN]')
        print("\nConfidence Intervals:")
        print("\nThrust Force vs Penetration Rate:")
        print(model1.conf_int())
        print("\nThrust Force vs Advance Rate:")
        print(model2.conf_int())
        create_polar_plot(df)
        visualize_parameters_vs_chainage(df)
        create_density_heatmap(df)
        visualize_torque_components(df)
        create_torque_vs_pressure_plot(df)

    rock_strength_file_path = "/content/drive/MyDrive/rock_strenght.xlsx"
    rock_df = read_rock_strength_data(rock_strength_file_path)

    if rock_df is not None:
        print("\nRock strength data:")
        print(rock_df.head(30))

        processed_rock_df = preprocess_rock_strength_data(rock_df)
        print("\nProcessed rock strength data:")
        print(processed_rock_df)

        create_rock_strength_comparison_chart(df, processed_rock_df)



        fig = create_thrust_force_vs_penetration_rate_plot(df, rock_df)

        if fig is not None:
            fig.show()
        else:
            print("Unable to create thrust force vs penetration rate plot.")
    else:
        print("Unable to create rock strength comparison chart due to missing data.")

    print("\nDataFrame Info:")
    df.info()
    print("\nDataFrame Summary:")
    print(df.describe())
    print("\nFirst few rows of the DataFrame:")
    print(df.head())

    print("\nUnique values in Arbeitsdruck column:")
    print(df['Working pressure [bar]'].unique())
    print("\nFirst few rows with new torque calculations:")
    print(df[['Working pressure [bar]', 'Revolution [rpm]', 'Calculated torque [kNm]']].head())
    print("\nFirst few rows with new advance rate calculations:")
    print(df[['Working pressure [bar]', 'Revolution [rpm]', 'Average Speed (mm/min)', 'Penetration_Rate']].mean())


if __name__ == "__main__":
    main()
