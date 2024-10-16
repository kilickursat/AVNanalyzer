import streamlit as st
import pandas as pd

# Chainage filtering and data averaging
def apply_chainage_filtering_and_averaging(df):
    st.sidebar.header("Chainage Filtering & Averaging")

    # Chainage range slider
    chainage_range = st.sidebar.slider("Select chainage range", 
                                       float(df['Chainage [mm]'].min()), 
                                       float(df['Chainage [mm]'].max()), 
                                       (float(df['Chainage [mm]'].min()), float(df['Chainage [mm]'].max())))

    # Filtering data based on chainage range
    filtered_df = df[(df['Chainage [mm]'] >= chainage_range[0]) & (df['Chainage [mm]'] <= chainage_range[1])]

    # Averaging based on sampling rate
    sampling_rate = st.sidebar.selectbox("Select sampling rate", ['Milliseconds', 'Seconds'])

    time_column = get_time_column(df)  # Assuming you already have a function to get time column

    if sampling_rate == 'Milliseconds':
        resampled_df = filtered_df.resample('S', on=time_column).mean()
    elif sampling_rate == 'Seconds':
        resampled_df = filtered_df.resample('T', on=time_column).mean()

    return resampled_df

# Manual input for cutting rings
def apply_cutting_ring_adjustment(df):
    st.sidebar.header("Cutting Ring Adjustment")

    # Number input for cutting rings
    cutting_rings = st.sidebar.number_input("Enter the number of cutting rings", min_value=1, value=1)

    # Adjust thrust force by cutting rings
    if 'Thrust Force [kN]' in df.columns:
        df['Thrust Force [kN]'] /= cutting_rings
    return df

# Penetration rate display (sensor-based or calculated)
def apply_penetration_rate_selection(df):
    st.sidebar.header("Penetration Rate Display")

    sensor_based_penetration_available = 'Sensor Penetration Rate' in df.columns

    # Radio button to select penetration rate type
    penetration_rate_type = st.radio(
        "Select penetration rate type",
        ['Sensor-based Penetration' if sensor_based_penetration_available else 'Calculated Penetration Rate'],
        index=0 if sensor_based_penetration_available else 1
    )

    if penetration_rate_type == 'Sensor-based Penetration':
        df['Penetration Rate [mm/rev]'] = df['Sensor Penetration Rate']
    else:
        df['Penetration Rate [mm/rev]'] = df['Advance Rate [mm/min]'] / df['Revolution [rpm]']
    
    return df
import pandas as pd
import streamlit as st
from datetime import datetime
import base64

# Improved CSV/XLSX handling
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=';', decimal=',', na_values=['', 'NA', 'N/A', 'nan', 'NaN'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Handle different time formats
def parse_time_column(df, time_column):
    # Parse time based on expected formats
    df['Parsed Time'] = pd.to_datetime(df[time_column], errors='coerce', infer_datetime_format=True)
    
    # Option for manual time format entry (if necessary)
    manual_format = st.sidebar.text_input("Enter time format manually (if needed)", value="%d/%m/%Y %I:%M:%S %p")
    if st.sidebar.button("Apply manual format"):
        df['Parsed Time'] = pd.to_datetime(df[time_column], format=manual_format, errors='coerce')
    
    return df

# Tunnel length calculation based on time and advance rate
def calculate_tunnel_length(df):
    if 'Parsed Time' not in df.columns:
        st.error("Time column not found or not parsed. Please ensure the time column is properly formatted.")
        return df

    # Calculate time difference and tunnel length
    df['Time Difference (s)'] = df['Parsed Time'].diff().dt.total_seconds().fillna(0)
    df['Tunnel Length [mm]'] = df['Time Difference (s)'] * df['Advance Rate [mm/min]'] / 60  # Convert to mm
    
    return df

# Main function that integrates the different parts without design changes
def main():
    st.title("Streamlit Data Processing App")

    uploaded_file = st.file_uploader("Upload CSV/Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file:
        # Load the data
        df = load_data(uploaded_file)

        if df is not None:
            # Apply chainage filtering and averaging
            df = apply_chainage_filtering_and_averaging(df)
            
            # Apply cutting ring adjustments
            df = apply_cutting_ring_adjustment(df)
            
            # Apply penetration rate selection
            df = apply_penetration_rate_selection(df)
            
            # Parse the time column
            time_column = get_time_column(df)  # Assuming you have a helper function to identify the time column
            df = parse_time_column(df, time_column)
            
            # Calculate tunnel length
            df = calculate_tunnel_length(df)
            
            # Display the processed DataFrame
            st.write(df)

            # Option to download the processed data
            if st.sidebar.button("Download Processed Data"):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV File</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)
                
if __name__ == '__main__':
    main()
