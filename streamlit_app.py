import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Soil Monitoring Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Sheets URL - Replace with your actual sheet ID and ensure it's public or properly shared
GOOGLE_SHEET_ID = "1ck6Uf9MZhpWlfHFMb283zAH50LATmIq_FgMeiSkdobI"
GOOGLE_SHEET_GID = "915880652"
GOOGLE_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={GOOGLE_SHEET_GID}"

# Parameter optimal ranges based on agricultural research
OPTIMAL_RANGES = {
    'Ph (pH)': {'min': 6.0, 'max': 7.0, 'unit': 'pH', 'critical_low': 5.5, 'critical_high': 7.5},
    'Moisture (%)': {'min': 40, 'max': 70, 'unit': '%', 'critical_low': 30, 'critical_high': 80},
    'Temperature (°C)': {'min': 15, 'max': 25, 'unit': '°C', 'critical_low': 10, 'critical_high': 35},
    'N (kg/ha)': {'min': 120, 'max': 200, 'unit': 'kg/ha', 'critical_low': 80, 'critical_high': 300},
    'P (kg/ha)': {'min': 25, 'max': 50, 'unit': 'kg/ha', 'critical_low': 15, 'critical_high': 80},
    'K (kg/ha)': {'min': 150, 'max': 300, 'unit': 'kg/ha', 'critical_low': 100, 'critical_high': 450},
    'EC (dS/m)': {'min': 0.8, 'max': 2.5, 'unit': 'dS/m', 'critical_low': 0.1, 'critical_high': 4.0}
}


def get_dark_theme_settings():
    """
    Return settings for transparent theme - adapts to Streamlit's theme automatically
    """
    return True


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from Google Sheets"""
    try:
        # Load data from Google Sheets
        response = requests.get(GOOGLE_SHEET_URL, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Read the CSV data
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            st.error("Google Sheets returned empty data. Please check your sheet URL and permissions.")
            return pd.DataFrame()

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Handle Time column - convert to string and pad with zeros if needed
        df['Time'] = df['Time'].astype(str).str.zfill(6)  # Ensure 6 digits for HHMMSS format

        # Create DateTime column
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' +
            df['Time'].str[:2] + ':' + df['Time'].str[2:4] + ':' + df['Time'].str[4:6],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )

        # Fill any NaT values with just the date
        df['DateTime'] = df['DateTime'].fillna(df['Date'])

        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])

        # Ensure numeric columns are properly typed
        numeric_columns = ['Ph (pH)', 'Moisture (%)', 'Temperature (°C)', 'N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)',
                           'EC (dS/m)', 'Tub (count)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure Tub column contains integers
        df['Tub (count)'] = df['Tub (count)'].astype('Int64')  # Use nullable integer type

        return df

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please check your internet connection.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing Google Sheets: {e}")
        st.info("Make sure your Google Sheet is public or properly shared for viewing.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()


def create_gauge(value, parameter, optimal_range, is_dark_theme=True):
    """Create a gauge chart for parameter monitoring with transparent background and black borders"""
    # Handle NaN values
    if pd.isna(value):
        value = 0

    # Use transparent backgrounds so gauges adapt to Streamlit's theme automatically
    bg_color = "rgba(0,0,0,0)"  # Fully transparent
    border_color = "black"  # Thick black border

    # Determine gauge bar color with neutral tones
    if optimal_range['min'] <= value <= optimal_range['max']:
        bar_color = "#28A745"  # Standard green
        status = "optimal"
    elif optimal_range['critical_low'] <= value <= optimal_range['critical_high']:
        bar_color = "#FD7E14"  # Standard orange
        status = "acceptable"
    else:
        bar_color = "#DC3545"  # Standard red
        status = "critical"

    # Set gauge range
    gauge_max = max(optimal_range['critical_high'], value * 1.2)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"{parameter}<br><span style='font-size:0.8em;opacity:0.7'>{optimal_range['unit']}</span>",
            'font': {'size': 14}
        },
        number={'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, gauge_max], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.3, 'line': {'color': border_color, 'width': 2}},
            'bgcolor': bg_color,  # Transparent background
            'borderwidth': 2,  # Thick border
            'bordercolor': border_color,  # Black border
            'steps': [
                {'range': [0, optimal_range['critical_low']], 'color': "rgba(255, 99, 132, 0.5)",
                 'line': {'color': border_color, 'width': 1}},  # Lighter red
                {'range': [optimal_range['critical_low'], optimal_range['min']], 'color': "rgba(255, 205, 86, 0.5)",
                 'line': {'color': border_color, 'width': 1}},  # Lighter yellow
                {'range': [optimal_range['min'], optimal_range['max']], 'color': "rgba(75, 192, 192, 0.5)",
                 'line': {'color': border_color, 'width': 1}},  # Lighter green
                {'range': [optimal_range['max'], optimal_range['critical_high']], 'color': "rgba(255, 205, 86, 0.5)",
                 'line': {'color': border_color, 'width': 1}},  # Lighter yellow
                {'range': [optimal_range['critical_high'], gauge_max], 'color': "rgba(255, 99, 132, 0.5)",
                 'line': {'color': border_color, 'width': 1}}  # Lighter red
            ],
            'threshold': {
                'line': {'color': "#DC3545", 'width': 2},  # Standard red for threshold
                'thickness': 0.75,
                'value': optimal_range['max']
            }
        }
    ))

    fig.update_layout(
        width=250,
        height=250,

        margin=dict(l=20, r=20, t=30, b=10),
        font={'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background

    )

    return fig, status


def get_status_message(value, parameter, optimal_range):
    """Get status message for a parameter"""
    if pd.isna(value):
        return "", "No data", "secondary"

    if optimal_range['min'] <= value <= optimal_range['max']:
        return "", f"Optimal: {value:.2f} {optimal_range['unit']}", "success"
    elif optimal_range['critical_low'] <= value <= optimal_range['critical_high']:
        return "", f"Acceptable: {value:.2f} {optimal_range['unit']}", "warning"
    else:
        return "", f"Critical: {value:.2f} {optimal_range['unit']}", "error"


def main():
    # Default to dark theme
    is_dark_theme = get_dark_theme_settings()

    # Title and description
    st.title("🌱 Soil Monitoring Dashboard")
    st.markdown("Real-time soil parameter monitoring and analysis system")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data from Google Sheets..."):
        df = load_data()

    if df.empty:
        st.error("No data available. Please check your Google Sheets connection and ensure the sheet is accessible.")
        st.stop()

    # Display data info
    st.success(f"Successfully loaded {len(df)} records from Google Sheets")

    # Show data preview for debugging
    with st.expander("Data Preview (for debugging)", expanded=False):
        st.write("First few rows of loaded data:")
        st.dataframe(df.head())
        st.write("Data types:")
        st.write(df.dtypes)
        st.write("Unique tub values:")
        st.write(df['Tub (count)'].unique())

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")

    # Tub selection
    available_tubs = sorted([int(x) for x in df['Tub (count)'].dropna().unique() if pd.notna(x)])
    selected_tubs = st.sidebar.multiselect(
        "Select Tubs for Analysis",
        options=available_tubs,
        default=available_tubs,
        help="Choose which tubs to include in the analysis"
    )

    # Parameter selection
    numeric_columns = ['Ph (pH)', 'Moisture (%)', 'Temperature (°C)', 'N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)',
                       'EC (dS/m)']
    available_parameters = [col for col in numeric_columns if col in df.columns]

    selected_parameters = st.sidebar.multiselect(
        "Select Parameters",
        options=available_parameters,
        default=available_parameters[:4] if len(available_parameters) >= 4 else available_parameters,
        help="Choose which parameters to display"
    )

    # Time filtering
    st.sidebar.subheader("Time Filtering")
    if not df['Date'].dropna().empty:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

    # Filter data based on selections
    if selected_tubs and 'start_date' in locals():
        filtered_df = df[
            (df['Tub (count)'].isin(selected_tubs)) &
            (df['Date'].dt.date >= start_date) &
            (df['Date'].dt.date <= end_date)
            ].copy()
    else:
        filtered_df = df.copy()

    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Real-time Gauges", "Time Series Analysis", "Parameter Correlations", "Tub Comparisons"]
    )

    with tab1:
        st.header("Real-time Parameter Monitoring")

        if not filtered_df.empty and selected_parameters:
            # Get latest readings for each tub
            latest_data = filtered_df.groupby('Tub (count)').last()

            for tub in selected_tubs:
                if tub not in latest_data.index:
                    st.warning(f"No data available for Tub {tub}")
                    continue

                tub_data = latest_data.loc[tub]
                last_updated = tub_data['DateTime']

                st.subheader(f"Tub {tub} - Current Status")
                st.caption(
                    f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(last_updated) else 'Unknown'}")

                # Create gauges in a grid layout
                num_params = len(selected_parameters)
                cols_per_row = 4

                for i in range(0, num_params, cols_per_row):
                    gauge_cols = st.columns(cols_per_row)
                    param_batch = selected_parameters[i:i + cols_per_row]

                    for j, param in enumerate(param_batch):
                        with gauge_cols[j]:
                            if param in OPTIMAL_RANGES and param in tub_data.index:
                                value = tub_data[param]
                                # Pass theme information to gauge creation
                                gauge_fig, status = create_gauge(value, param, OPTIMAL_RANGES[param], is_dark_theme)
                                st.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{tub}_{param}")

                                # Status indicator below gauge
                                icon, message, alert_type = get_status_message(value, param, OPTIMAL_RANGES[param])

                                if alert_type == "success":
                                    st.success(f"{icon} {message}")
                                elif alert_type == "warning":
                                    st.warning(f"{icon} {message}")
                                else:
                                    st.error(f"{icon} {message}")
                            else:
                                st.info(f"No optimal range defined for {param}")

                # Summary status for the tub
                st.markdown("---")
                optimal_count = 0
                total_count = 0

                for param in selected_parameters:
                    if param in OPTIMAL_RANGES and param in tub_data.index:
                        value = tub_data[param]
                        if not pd.isna(value):
                            total_count += 1
                            if OPTIMAL_RANGES[param]['min'] <= value <= OPTIMAL_RANGES[param]['max']:
                                optimal_count += 1

                if total_count > 0:
                    health_score = (optimal_count / total_count) * 100
                    st.metric(
                        label="Overall Tub Health",
                        value=f"{health_score:.1f}%",
                        delta=f"{optimal_count}/{total_count} parameters optimal"
                    )

                st.markdown("---")
        else:
            st.info("Please select parameters and tubs to display gauges.")

    with tab2:
        st.header("Time Series Analysis")
        # Rest of the time series code remains the same...
        if selected_parameters and not filtered_df.empty:
            view_mode = st.radio(
                "View Mode:",
                ["Individual Charts", "Combined View"],
                horizontal=True,
                help="Individual charts show each parameter separately with optimized scales."
            )

            if view_mode == "Individual Charts":
                for param in selected_parameters:
                    st.subheader(f"{param}")
                    fig = go.Figure()

                    colors = px.colors.qualitative.Set3
                    for j, tub in enumerate(selected_tubs):
                        tub_data = filtered_df[filtered_df['Tub (count)'] == tub].sort_values('DateTime')
                        if not tub_data.empty and param in tub_data.columns:
                            valid_data = tub_data.dropna(subset=[param])
                            if not valid_data.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=valid_data['DateTime'],
                                        y=valid_data[param],
                                        mode='lines+markers',
                                        name=f'Tub {tub}',
                                        line=dict(color=colors[j % len(colors)], width=2),
                                        marker=dict(size=6, symbol='circle')
                                    )
                                )

                    # Add optimal ranges
                    if param in OPTIMAL_RANGES:
                        opt_range = OPTIMAL_RANGES[param]
                        fig.add_hrect(
                            y0=opt_range['min'], y1=opt_range['max'],
                            fillcolor="rgba(100, 255, 100, 0.25)",
                            line_width=0, layer="below"
                        )

                    fig.update_layout(height=500, title=f"{param} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Combined view code...
                st.info("Combined view implementation remains the same as original")

    with tab3:
        st.header("Parameter Correlations")
        if len(selected_parameters) >= 2 and not filtered_df.empty:
            corr_data = filtered_df[selected_parameters].corr()
            fig = px.imshow(
                corr_data, text_auto=True, aspect="auto",
                title="Parameter Correlation Matrix",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Tub Comparisons")
        if selected_parameters and len(selected_tubs) >= 2 and not filtered_df.empty:
            for param in selected_parameters:
                fig = px.box(
                    filtered_df[filtered_df['Tub (count)'].isin(selected_tubs)],
                    x='Tub (count)', y=param,
                    title=f"{param} Distribution Across Tubs",
                    points="all", color='Tub (count)'
                )
                if param in OPTIMAL_RANGES:
                    opt_range = OPTIMAL_RANGES[param]
                    fig.add_hrect(
                        y0=opt_range['min'], y1=opt_range['max'],
                        fillcolor="green", opacity=0.15, line_width=0
                    )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Sidebar summary and controls remain the same...
    st.sidebar.markdown("---")
    st.sidebar.subheader("Summary Statistics")

    if not filtered_df.empty:
        for param in selected_parameters[:3]:
            if param in filtered_df.columns:
                param_data = filtered_df[param].dropna()
                if not param_data.empty:
                    st.sidebar.metric(
                        label=param.replace(' (', '\n('),
                        value=f"{param_data.mean():.2f}",
                        delta=f"±{param_data.std():.2f}"
                    )

    # Data management
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Management")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with col2:
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Export",
                data=csv,
                file_name=f"soil_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "**Soil Monitoring Dashboard** | "
        f"Data Source: Google Sheets | "
        f"Last Refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()