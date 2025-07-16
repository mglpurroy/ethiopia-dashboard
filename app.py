import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
import datetime
warnings.filterwarnings('ignore')

# Try to import geospatial libraries with fallbacks
try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
    GEOSPATIAL_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Geospatial libraries not available: {e}")
    st.info("Run: `conda install -c conda-forge geopandas rasterio pyproj fiona shapely gdal`")
    GEOSPATIAL_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🇪🇹 Ethiopia Violence Analysis Dashboard",
    page_icon="🇪🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding: 0.5rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    .stMetric > div {
        color: white !important;
    }
    .stMetric label {
        color: white !important;
        font-weight: 600;
    }
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .status-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border: 2px solid #667eea;
        border-radius: 0.8rem;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DATA_PATH = Path("data/")
PROCESSED_PATH = DATA_PATH / "processed"
START_YEAR = 2009
END_YEAR = 2025

# Sample data for demonstration when geospatial libraries are not available
SAMPLE_DATA = {
    'admin1': [
        {'pcode': 'ET01', 'name': 'Tigray', 'pop': 5450000, 'deaths': 1250, 'woredas_total': 35, 'woredas_affected': 8, 'lat': 14.2, 'lng': 38.5},
        {'pcode': 'ET02', 'name': 'Afar', 'pop': 1812000, 'deaths': 89, 'woredas_total': 29, 'woredas_affected': 3, 'lat': 11.7, 'lng': 40.9},
        {'pcode': 'ET03', 'name': 'Amhara', 'pop': 21140000, 'deaths': 892, 'woredas_total': 140, 'woredas_affected': 25, 'lat': 11.5, 'lng': 37.8},
        {'pcode': 'ET04', 'name': 'Oromia', 'pop': 35467000, 'deaths': 1456, 'woredas_total': 287, 'woredas_affected': 45, 'lat': 8.5, 'lng': 39.5},
        {'pcode': 'ET05', 'name': 'Somali', 'pop': 5441000, 'deaths': 234, 'woredas_total': 93, 'woredas_affected': 12, 'lat': 6.5, 'lng': 43.5},
        {'pcode': 'ET06', 'name': 'SNNP', 'pop': 15042000, 'deaths': 567, 'woredas_total': 133, 'woredas_affected': 18, 'lat': 6.8, 'lng': 37.5}
    ],
    'admin2': [
        {'pcode': 'ET0101', 'name': 'Central Tigray', 'parent': 'Tigray', 'pop': 1200000, 'deaths': 450, 'woredas_total': 8, 'woredas_affected': 5, 'lat': 14.3, 'lng': 38.8},
        {'pcode': 'ET0102', 'name': 'Eastern Tigray', 'parent': 'Tigray', 'pop': 980000, 'deaths': 234, 'woredas_total': 6, 'woredas_affected': 2, 'lat': 14.1, 'lng': 39.3},
        {'pcode': 'ET0301', 'name': 'North Wollo', 'parent': 'Amhara', 'pop': 1850000, 'deaths': 234, 'woredas_total': 12, 'woredas_affected': 6, 'lat': 12.1, 'lng': 38.9},
        {'pcode': 'ET0302', 'name': 'South Wollo', 'parent': 'Amhara', 'pop': 2100000, 'deaths': 189, 'woredas_total': 15, 'woredas_affected': 4, 'lat': 10.8, 'lng': 38.7},
        {'pcode': 'ET0401', 'name': 'West Arsi', 'parent': 'Oromia', 'pop': 2200000, 'deaths': 345, 'woredas_total': 18, 'woredas_affected': 8, 'lat': 7.5, 'lng': 38.5},
        {'pcode': 'ET0402', 'name': 'East Shewa', 'parent': 'Oromia', 'pop': 1950000, 'deaths': 123, 'woredas_total': 14, 'woredas_affected': 3, 'lat': 8.8, 'lng': 39.2}
    ]
}

# Create sample time series data
def create_sample_time_series():
    """Create sample time series data for demonstration"""
    years = list(range(START_YEAR, END_YEAR + 1))
    data = []
    
    for year in years:
        # Simulate some trends
        base_violence = 0.1 + 0.02 * np.sin((year - START_YEAR) * 0.5)
        noise = np.random.normal(0, 0.01)
        share_zones = max(0, min(1, base_violence + noise))
        
        total_zones = 85
        above_threshold = int(total_zones * share_zones)
        total_woredas = 850
        violence_affected = int(total_woredas * share_zones * 0.3)
        deaths = int(1000 + 500 * share_zones + np.random.normal(0, 100))
        population = 120000000
        
        data.append({
            'year': year,
            'above_threshold': above_threshold,
            'total_zones': total_zones,
            'violence_affected': violence_affected,
            'total_woredas': total_woredas,
            'ACLED_BRD_total': max(0, deaths),
            'pop_count': population,
            'share_zones_above_threshold': share_zones,
            'share_woredas_affected': violence_affected / total_woredas,
            'death_rate_national': (deaths / population) * 1e5
        })
    
    return pd.DataFrame(data)

def load_conflict_data_fallback():
    """Load ACLED data or use sample data"""
    try:
        if PROCESSED_PATH.exists() and (PROCESSED_PATH / "intersection_result_acled.csv").exists():
            acled_data = pd.read_csv(PROCESSED_PATH / "intersection_result_acled.csv")
            ethiopia_acled = acled_data[acled_data['GID_0'] == 'ETH'].copy()
            return ethiopia_acled
        else:
            st.warning("ACLED data file not found. Using sample data for demonstration.")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load ACLED data: {e}. Using sample data.")
        return pd.DataFrame()

def create_sample_map(agg_level, agg_thresh, year):
    """Create map using sample data"""
    m = folium.Map(location=[9.15, 40.49], zoom_start=6, tiles='OpenStreetMap')
    
    # Use sample data based on admin level
    data = SAMPLE_DATA['admin1'] if agg_level == 'ADM1' else SAMPLE_DATA['admin2']
    
    for unit in data:
        share_affected = unit['woredas_affected'] / unit['woredas_total']
        is_high_violence = share_affected > agg_thresh
        
        if is_high_violence:
            color = '#d73027'
            status = 'HIGH VIOLENCE'
        elif share_affected > 0:
            color = '#fd8d3c'
            status = 'Some Violence'
        else:
            color = '#2c7fb8'
            status = 'Low/No Violence'
        
        popup_content = f"""
        <div style="font-family: Arial, sans-serif; min-width: 250px;">
            <h3 style="color: {color}; margin: 0 0 10px 0;">{unit['name']}</h3>
            <div style="background: {color}; color: white; padding: 5px; border-radius: 3px; text-align: center; margin-bottom: 10px;">
                <strong>{status}</strong>
            </div>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>Share Affected:</strong></td><td style="text-align: right;">{share_affected:.1%}</td></tr>
                <tr><td><strong>Affected Woredas:</strong></td><td style="text-align: right;">{unit['woredas_affected']}/{unit['woredas_total']}</td></tr>
                <tr><td><strong>Population:</strong></td><td style="text-align: right;">{unit['pop']:,}</td></tr>
                <tr><td><strong>Total Deaths:</strong></td><td style="text-align: right;">{unit['deaths']:,}</td></tr>
            </table>
        </div>
        """
        
        folium.CircleMarker(
            location=[unit['lat'], unit['lng']],
            radius=max(5, np.sqrt(unit['pop'] / 100000) * 3),
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{unit['name']}: {share_affected:.1%} ({status})",
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def create_time_series_charts(yearly_summary):
    """Create time series visualization charts"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'High-Violence Zones Over Time',
            'National Violence Death Rate',
            'Total Deaths vs Affected Woredas',
            'Trends Summary'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Plot 1: High-violence zones over time
    fig.add_trace(
        go.Scatter(
            x=yearly_summary['year'],
            y=yearly_summary['share_zones_above_threshold'] * 100,
            mode='lines+markers',
            name='High-Violence Zones %',
            line=dict(color='#d73027', width=3),
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(215, 48, 39, 0.3)'
        ),
        row=1, col=1
    )
    
    # Plot 2: National death rate
    fig.add_trace(
        go.Bar(
            x=yearly_summary['year'],
            y=yearly_summary['death_rate_national'],
            name='Death Rate (per 100k)',
            marker_color='#fd8d3c',
            opacity=0.8
        ),
        row=1, col=2
    )
    
    # Plot 3: Total deaths and affected woredas
    fig.add_trace(
        go.Scatter(
            x=yearly_summary['year'],
            y=yearly_summary['ACLED_BRD_total'],
            mode='lines+markers',
            name='Total Deaths',
            line=dict(color='#2c7fb8', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1,
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_summary['year'],
            y=yearly_summary['violence_affected'],
            mode='lines+markers',
            name='Affected Woredas',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1,
        secondary_y=True
    )
    
    # Plot 4: Summary metrics
    fig.add_trace(
        go.Scatter(
            x=yearly_summary['year'],
            y=yearly_summary['share_woredas_affected'] * 100,
            mode='lines+markers',
            name='Woredas Affected %',
            line=dict(color='#9467bd', width=2),
            marker=dict(size=6)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Ethiopia Violence Trends Over Time (Sample Data)",
        title_font_size=16,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_yaxes(title_text="Share of Zones (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Deaths per 100,000", row=1, col=2)
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Total Deaths", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Affected Woredas", row=2, col=1, secondary_y=True)
    
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Woredas Affected (%)", row=2, col=2)
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🇪🇹 Ethiopia Violence Analysis Dashboard</h1>
        <p class="subtitle">Interactive map-based analysis of violence patterns across Ethiopian administrative units</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for geospatial libraries
    if not GEOSPATIAL_AVAILABLE:
        st.warning("⚠️ **Geospatial libraries not fully available.** Running in demo mode with sample data.")
        st.info("""
        **To use real data, install geospatial dependencies:**
        
        ```bash
        conda install -c conda-forge geopandas rasterio pyproj fiona shapely gdal
        ```
        
        Then restart the application.
        """)
    
    # Sidebar controls
    st.sidebar.title("🎛️ Analysis Controls")
    
    analysis_mode = st.sidebar.selectbox(
        "📊 Analysis Mode",
        ["🗺️ Interactive Map Dashboard", "📈 Time Series Analysis", "🔄 Combined Analysis"]
    )
    
    if analysis_mode in ["🗺️ Interactive Map Dashboard", "🔄 Combined Analysis"]:
        st.sidebar.markdown("### 📊 Violence Classification")
        rate_thresh = st.sidebar.slider("Death Rate Threshold (per 100k)", 0.5, 20.0, 4.0, 0.5)
        abs_thresh = st.sidebar.slider("Min Deaths Threshold", 1, 100, 20, 1)
        agg_thresh = st.sidebar.slider("Aggregation Threshold", 0.05, 0.5, 0.2, 0.05)
        
        st.sidebar.markdown("### 🗺️ Display Settings")
        year = st.sidebar.selectbox("Analysis Year", list(range(START_YEAR, END_YEAR + 1)), index=14)
        admin_level = st.sidebar.selectbox(
            "Administrative Level",
            ["ADM1", "ADM2"],
            format_func=lambda x: "Admin 1 (Regions)" if x == "ADM1" else "Admin 2 (Zones)",
            index=1
        )
        map_variable = st.sidebar.selectbox(
            "Map Variable",
            ["share_woredas", "share_population"],
            format_func=lambda x: "Share of Woredas Affected" if x == "share_woredas" else "Share of Population Affected"
        )
        
        # Status info
        admin_level_text = "Admin 1 (Regions)" if admin_level == "ADM1" else "Admin 2 (Zones)"
        map_var_text = "Share of Woredas Affected" if map_variable == "share_woredas" else "Share of Population Affected"
        
        st.markdown(f"""
        <div class="status-info">
            <strong>📊 Current Analysis Configuration</strong><br>
            <strong>Year:</strong> {year} | 
            <strong>Level:</strong> {admin_level_text} | 
            <strong>Map Variable:</strong> {map_var_text}<br>
            <strong>Woreda Classification:</strong> Death rate >{rate_thresh:.1f}/100k AND >{abs_thresh} deaths<br>
            <strong>Unit Threshold:</strong> >{agg_thresh:.1%} of woredas affected → Marked as high-violence area
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive Map Dashboard
    if analysis_mode in ["🗺️ Interactive Map Dashboard", "🔄 Combined Analysis"]:
        st.markdown("## 🗺️ Interactive Violence Map")
        
        # Create map and metrics in columns
        map_col, metrics_col = st.columns([3, 1])
        
        with map_col:
            if GEOSPATIAL_AVAILABLE:
                st.info("Loading real geospatial data...")
                # Here you would call your actual data loading functions
                st.info("Real data loading would happen here with your actual functions.")
            else:
                # Use sample data
                map_obj = create_sample_map(admin_level, agg_thresh, year)
                st_folium(map_obj, width=700, height=600)
        
        with metrics_col:
            st.markdown("### 📊 Quick Stats")
            
            # Sample metrics
            data = SAMPLE_DATA['admin1'] if admin_level == 'ADM1' else SAMPLE_DATA['admin2']
            
            total_units = len(data)
            above_threshold_count = sum(1 for unit in data if (unit['woredas_affected'] / unit['woredas_total']) > agg_thresh)
            total_woredas = sum(unit['woredas_total'] for unit in data)
            affected_woredas = sum(unit['woredas_affected'] for unit in data)
            total_population = sum(unit['pop'] for unit in data)
            total_deaths = sum(unit['deaths'] for unit in data)
            
            st.metric("🎯 High Violence Units", f"{above_threshold_count}/{total_units}", f"{above_threshold_count/total_units*100:.1f}%")
            st.metric("🏘️ Affected Woredas", f"{affected_woredas}/{total_woredas}", f"{affected_woredas/total_woredas*100:.1f}%")
            st.metric("👥 Total Population", f"{total_population:,.0f}", "Sample data")
            st.metric("💀 Total Deaths", f"{total_deaths:,}", f"in {year}")
        
        # Supporting charts
        st.markdown("## 📈 Supporting Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("🎯 Violence-Affected Areas")
            
            # Create sample bar chart
            df_sample = pd.DataFrame(data)
            df_sample['share_affected'] = df_sample['woredas_affected'] / df_sample['woredas_total']
            df_sample = df_sample[df_sample['share_affected'] > 0].sort_values('share_affected')
            
            if len(df_sample) > 0:
                colors = ['#d73027' if share > agg_thresh else '#fd8d3c' for share in df_sample['share_affected']]
                
                fig_bar = go.Figure(data=go.Bar(
                    y=df_sample['name'],
                    x=df_sample['share_affected'] * 100,
                    orientation='h',
                    marker_color=colors
                ))
                
                fig_bar.add_vline(x=agg_thresh*100, line_dash="dash", line_color="red")
                fig_bar.update_layout(title="Ranked by Woreda Share", xaxis_title="Share of Woredas Affected (%)", height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with chart_col2:
            st.subheader("📊 Distribution of Violence Levels")
            
            no_violence = sum(1 for unit in data if unit['woredas_affected'] == 0)
            some_violence = sum(1 for unit in data if unit['woredas_affected'] > 0 and (unit['woredas_affected'] / unit['woredas_total']) <= agg_thresh)
            high_violence = sum(1 for unit in data if (unit['woredas_affected'] / unit['woredas_total']) > agg_thresh)
            
            fig_pie = go.Figure(data=go.Pie(
                labels=['No Violence', 'Some Violence', 'High Violence'],
                values=[no_violence, some_violence, high_violence],
                marker_colors=['#2c7fb8', '#fd8d3c', '#d73027']
            ))
            
            fig_pie.update_layout(title=f"{len(data)} {admin_level_text.split()[1]} Total", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Time Series Analysis
    if analysis_mode in ["📈 Time Series Analysis", "🔄 Combined Analysis"]:
        st.markdown("## 📈 Time Series Analysis (2009-2025)")
        
        # Create sample time series data
        yearly_summary = create_sample_time_series()
        
        # Display charts
        fig_ts = create_time_series_charts(yearly_summary)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Summary table
        st.subheader("📊 Yearly Summary Statistics")
        
        display_summary = yearly_summary[['year', 'above_threshold', 'total_zones', 'violence_affected', 'total_woredas', 'ACLED_BRD_total', 'death_rate_national']].copy()
        display_summary.columns = ['Year', 'High-Violence Zones', 'Total Zones', 'Affected Woredas', 'Total Woredas', 'Deaths', 'Death Rate (per 100k)']
        
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
        🇪🇹 Ethiopia Violence Analysis Dashboard | Built with Streamlit<br>
        📊 Data sources: ACLED, WorldPop, Ethiopian Central Statistical Agency<br>
        ⚠️ Currently running with sample data for demonstration
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()