import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import folium
from streamlit_folium import st_folium
from pathlib import Path
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import io
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ethiopia Violence Analysis Dashboard",
    page_icon="🇪🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        text-align: center;
    }
    .status-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Data paths - adjust these for your deployment
DATA_PATH = Path("data/")
PROCESSED_PATH = DATA_PATH / "processed"
POPULATION_RASTER = DATA_PATH / "eth_ppp_2020.tif"
ADMIN_SHAPEFILES = {
    1: DATA_PATH / "eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm1_csa_bofedb_2021.shp",
    2: DATA_PATH / "eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm2_csa_bofedb_2021.shp",
    3: DATA_PATH / "eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm3_csa_bofedb_2021.shp"
}

START_YEAR = 2009
END_YEAR = 2025

@st.cache_data
def generate_12_month_periods():
    """Generate 12-month periods every 6 months"""
    periods = []
    
    # Calendar year periods (Jan-Dec)
    for year in range(START_YEAR, END_YEAR + 1):
        periods.append({
            'label': f'Jan {year} - Dec {year}',
            'start_month': 1,
            'start_year': year,
            'end_month': 12,
            'end_year': year,
            'type': 'calendar'
        })
    
    # Mid-year periods (Jul-Jun)
    for year in range(START_YEAR, END_YEAR):
        periods.append({
            'label': f'Jul {year} - Jun {year+1}',
            'start_month': 7,
            'start_year': year,
            'end_month': 6,
            'end_year': year + 1,
            'type': 'mid_year'
        })
    
    return periods

@st.cache_data
def load_population_data():
    """Load and cache population data with better error handling"""
    try:
        # Check if shapefile exists
        if not ADMIN_SHAPEFILES[3].exists():
            st.error(f"Shapefile not found: {ADMIN_SHAPEFILES[3]}")
            return pd.DataFrame()
        
        # Check if raster exists
        if not POPULATION_RASTER.exists():
            st.error(f"Population raster not found: {POPULATION_RASTER}")
            return pd.DataFrame()
        
        # Load shapefile
        admin3_gdf = gpd.read_file(ADMIN_SHAPEFILES[3])
        
        # Check if raster processing is needed
        try:
            with rasterio.open(POPULATION_RASTER) as src:
                # Check CRS compatibility
                if admin3_gdf.crs != src.crs:
                    try:
                        admin3_gdf = admin3_gdf.to_crs(src.crs)
                    except Exception as crs_error:
                        st.warning(f"CRS transformation failed: {crs_error}. Using original CRS.")
                
                population_data = []
                progress_bar = st.progress(0)
                total_rows = len(admin3_gdf)
                
                for idx, row in admin3_gdf.iterrows():
                    try:
                        geom = [row.geometry.__geo_interface__]
                        out_image, _ = mask(src, geom, crop=True, nodata=0)
                        pop_sum = out_image[out_image > 0].sum()
                        
                        population_data.append({
                            'ADM3_PCODE': row['ADM3_PCODE'],
                            'ADM3_EN': row['ADM3_EN'],
                            'ADM2_PCODE': row['ADM2_PCODE'],
                            'ADM2_EN': row['ADM2_EN'],
                            'ADM1_PCODE': row['ADM1_PCODE'],
                            'ADM1_EN': row['ADM1_EN'],
                            'ADM0_PCODE': row['ADM0_PCODE'],
                            'pop_count': int(pop_sum),
                            'pop_count_millions': pop_sum / 1e6
                        })
                    except Exception as row_error:
                        # Skip problematic rows but continue processing
                        continue
                    
                    # Update progress
                    if idx % 50 == 0:  # Update every 50 rows
                        progress_bar.progress((idx + 1) / total_rows)
                
                progress_bar.empty()
                
                if not population_data:
                    st.error("No population data could be extracted from raster.")
                    return pd.DataFrame()
                
                return pd.DataFrame(population_data)
                
        except Exception as raster_error:
            st.error(f"Error processing population raster: {raster_error}")
            # Try to load a simplified version without raster processing
            st.warning("Attempting to load simplified data without population raster...")
            
            # Create dummy population data based on admin units
            simplified_data = []
            for _, row in admin3_gdf.iterrows():
                simplified_data.append({
                    'ADM3_PCODE': row['ADM3_PCODE'],
                    'ADM3_EN': row['ADM3_EN'],
                    'ADM2_PCODE': row['ADM2_PCODE'],
                    'ADM2_EN': row['ADM2_EN'],
                    'ADM1_PCODE': row['ADM1_PCODE'],
                    'ADM1_EN': row['ADM1_EN'],
                    'ADM0_PCODE': row['ADM0_PCODE'],
                    'pop_count': 50000,  # Default population estimate
                    'pop_count_millions': 0.05
                })
            
            st.info("Using simplified population estimates. Results may be less accurate.")
            return pd.DataFrame(simplified_data)
            
    except Exception as e:
        st.error(f"Error loading population data: {str(e)}")
        st.info("Troubleshooting tips:")
        st.info("1. Install pyproj: pip install pyproj")
        st.info("2. Try conda: conda install -c conda-forge pyproj")
        st.info("3. Check data file paths")
        return pd.DataFrame()

@st.cache_data
def load_conflict_data():
    """Load and cache conflict data"""
    try:
        acled_data = pd.read_csv(PROCESSED_PATH / "intersection_result_acled.csv")
        ethiopia_acled = acled_data[acled_data['GID_0'] == 'ETH'].copy()
        
        # Map GID codes to ADM_PCODE system
        conflict_mapped = ethiopia_acled.groupby(['year', 'month', 'GID_3']).agg({
            'ACLED_BRD_state': 'sum',
            'ACLED_BRD_nonstate': 'sum',
            'ACLED_BRD_total': 'sum'
        }).reset_index()
        
        # Rename GID_3 to ADM3_PCODE for consistency
        conflict_mapped = conflict_mapped.rename(columns={'GID_3': 'ADM3_PCODE'})
        
        return conflict_mapped
    except Exception as e:
        st.error(f"Error loading conflict data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_admin_boundaries():
    """Load administrative boundaries with better error handling"""
    boundaries = {}
    for level in [1, 2, 3]:
        try:
            if not ADMIN_SHAPEFILES[level].exists():
                st.error(f"Shapefile not found: {ADMIN_SHAPEFILES[level]}")
                boundaries[level] = gpd.GeoDataFrame()
                continue
                
            gdf = gpd.read_file(ADMIN_SHAPEFILES[level])
            boundaries[level] = gdf
            st.success(f"✅ Loaded Admin {level} boundaries: {len(gdf)} features")
            
        except Exception as e:
            st.error(f"Error loading admin {level} boundaries: {str(e)}")
            boundaries[level] = gpd.GeoDataFrame()
    
    return boundaries

def filter_data_by_period(data, period_info):
    """Filter data based on 12-month period"""
    if len(data) == 0:
        return data
    
    start_year = period_info['start_year']
    end_year = period_info['end_year']
    start_month = period_info['start_month']
    end_month = period_info['end_month']
    
    if period_info['type'] == 'calendar':
        # Calendar year: Jan-Dec
        return data[(data['year'] == start_year) & 
                   (data['month'] >= start_month) & 
                   (data['month'] <= end_month)]
    else:
        # Mid-year: Jul-Jun
        return data[((data['year'] == start_year) & (data['month'] >= start_month)) |
                   ((data['year'] == end_year) & (data['month'] <= end_month))]

def classify_and_aggregate_data(pop_data, conflict_data, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level):
    """Classify woredas and aggregate to selected administrative level"""
    
    # Filter conflict data for selected period
    period_conflict = filter_data_by_period(conflict_data, period_info)
    
    # Aggregate conflict data by woreda for the period
    if len(period_conflict) > 0:
        conflict_agg = period_conflict.groupby('ADM3_PCODE').agg({
            'ACLED_BRD_state': 'sum',
            'ACLED_BRD_nonstate': 'sum',
            'ACLED_BRD_total': 'sum'
        }).reset_index()
    else:
        conflict_agg = pd.DataFrame()
    
    # Merge with population data
    if len(conflict_agg) > 0:
        merged = pd.merge(pop_data, conflict_agg, on='ADM3_PCODE', how='left').fillna(0)
    else:
        merged = pop_data.copy()
        for col in ['ACLED_BRD_state', 'ACLED_BRD_nonstate', 'ACLED_BRD_total']:
            merged[col] = 0
    
    # Calculate death rates
    merged['acled_total_death_rate'] = (merged['ACLED_BRD_total'] / (merged['pop_count_millions'] * 1e6)) * 1e5
    
    # Classify woredas
    merged['violence_affected'] = (
        (merged['acled_total_death_rate'] > rate_thresh) & 
        (merged['ACLED_BRD_total'] > abs_thresh)
    )
    
    # Aggregate to selected level
    if agg_level == 'ADM1':
        group_cols = ['ADM1_PCODE', 'ADM1_EN']
    else:  # ADM2
        group_cols = ['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN']
    
    aggregated = merged.groupby(group_cols).agg({
        'pop_count': 'sum',
        'violence_affected': 'sum',
        'ADM3_PCODE': 'count',
        'ACLED_BRD_total': 'sum'
    }).reset_index()
    
    aggregated.rename(columns={'ADM3_PCODE': 'total_woredas'}, inplace=True)
    
    # Calculate shares
    aggregated['share_woredas_affected'] = aggregated['violence_affected'] / aggregated['total_woredas']
    
    # Calculate population share
    affected_pop = merged[merged['violence_affected']].groupby(group_cols[0])['pop_count'].sum().reset_index()
    affected_pop.rename(columns={'pop_count': 'affected_population'}, inplace=True)
    aggregated = pd.merge(aggregated, affected_pop, on=group_cols[0], how='left').fillna(0)
    aggregated['share_population_affected'] = aggregated['affected_population'] / aggregated['pop_count']
    
    # Mark units above threshold
    aggregated['above_threshold'] = aggregated['share_woredas_affected'] > agg_thresh
    
    return aggregated, merged

def create_admin_map(aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh):
    """Create administrative units map"""
    
    # Determine columns
    pcode_col = f'{agg_level}_PCODE'
    name_col = f'{agg_level}_EN'
    
    if map_var == 'share_woredas':
        value_col = 'share_woredas_affected'
        value_label = 'Share of Woredas Affected'
    else:
        value_col = 'share_population_affected'
        value_label = 'Share of Population Affected'
    
    # Get appropriate boundary data
    map_level_num = 1 if agg_level == 'ADM1' else 2
    gdf = boundaries[map_level_num]
    
    # Merge data with boundaries
    merged_gdf = gdf.merge(
        aggregated[[pcode_col, value_col, 'above_threshold', 'violence_affected', 'total_woredas', 'pop_count', 'ACLED_BRD_total']], 
        on=pcode_col, 
        how='left'
    ).fillna({
        value_col: 0, 
        'above_threshold': False, 
        'violence_affected': 0, 
        'total_woredas': 0,
        'pop_count': 0,
        'ACLED_BRD_total': 0
    })
    
    # Create map
    m = folium.Map(location=[9.15, 40.49], zoom_start=6, tiles='OpenStreetMap')
    
    # Add choropleth layer
    for _, row in merged_gdf.iterrows():
        value = row[value_col]
        
        if value > agg_thresh:
            color = '#d73027'  # Red - High violence
            opacity = 0.8
            status = "HIGH VIOLENCE"
        elif value > 0:
            color = '#fd8d3c'  # Orange - Some violence
            opacity = 0.7
            status = "Some Violence"
        else:
            color = '#2c7fb8'  # Blue - Low/No violence
            opacity = 0.4
            status = "Low/No Violence"
        
        popup_content = f"""
        <div style="width: 300px; font-family: Arial, sans-serif;">
            <h3 style="color: {color}; margin: 0 0 10px 0;">{row.get(name_col, 'Unknown')}</h3>
            <div style="background: {color}; color: white; padding: 5px; border-radius: 3px; text-align: center; margin-bottom: 10px;">
                <strong>{status}</strong>
            </div>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>{value_label}:</strong></td><td style="text-align: right;">{row[value_col]:.1%}</td></tr>
                <tr><td><strong>Above Threshold:</strong></td><td style="text-align: right;">{'Yes' if row['above_threshold'] else 'No'}</td></tr>
                <tr><td><strong>Affected Woredas:</strong></td><td style="text-align: right;">{row['violence_affected']}/{row['total_woredas']}</td></tr>
                <tr><td><strong>Population:</strong></td><td style="text-align: right;">{row['pop_count']:,.0f}</td></tr>
                <tr><td><strong>Total Deaths:</strong></td><td style="text-align: right;">{row['ACLED_BRD_total']:,.0f}</td></tr>
            </table>
        </div>
        """
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=350),
            tooltip=f"{row.get(name_col, 'Unknown')}: {row[value_col]:.1%} ({status})"
        ).add_to(m)
    
    return m

def create_woreda_map(woreda_data, boundaries, period_info, rate_thresh, abs_thresh):
    """Create woreda classification map"""
    
    # Get woreda boundaries
    woreda_gdf = boundaries[3]
    
    # Merge with classification data
    merged_woreda = woreda_gdf.merge(
        woreda_data[['ADM3_PCODE', 'violence_affected', 'ACLED_BRD_total', 'acled_total_death_rate']], 
        on='ADM3_PCODE', 
        how='left'
    ).fillna({
        'violence_affected': False, 
        'ACLED_BRD_total': 0,
        'acled_total_death_rate': 0
    })
    
    # Create map
    m = folium.Map(location=[9.15, 40.49], zoom_start=6, tiles='OpenStreetMap')
    
    # Add woreda layer
    for _, row in merged_woreda.iterrows():
        if row['violence_affected']:
            color = '#d73027'  # Red - Violence affected
            opacity = 0.8
            status = "VIOLENCE AFFECTED"
        elif row['ACLED_BRD_total'] > 0:
            color = '#fd8d3c'  # Orange - Some deaths but below threshold
            opacity = 0.6
            status = "Below Threshold"
        else:
            color = '#2c7fb8'  # Blue - No violence
            opacity = 0.3
            status = "No Violence"
        
        popup_content = f"""
        <div style="width: 280px; font-family: Arial, sans-serif;">
            <h3 style="color: {color}; margin: 0 0 10px 0;">{row.get('ADM3_EN', 'Unknown')}</h3>
            <div style="background: {color}; color: white; padding: 5px; border-radius: 3px; text-align: center; margin-bottom: 10px;">
                <strong>{status}</strong>
            </div>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>Zone:</strong></td><td style="text-align: right;">{row.get('ADM2_EN', 'Unknown')}</td></tr>
                <tr><td><strong>Region:</strong></td><td style="text-align: right;">{row.get('ADM1_EN', 'Unknown')}</td></tr>
                <tr><td><strong>Total Deaths:</strong></td><td style="text-align: right;">{row['ACLED_BRD_total']:,.0f}</td></tr>
                <tr><td><strong>Death Rate:</strong></td><td style="text-align: right;">{row['acled_total_death_rate']:.1f}/100k</td></tr>
            </table>
        </div>
        """
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=320),
            tooltip=f"{row.get('ADM3_EN', 'Unknown')}: {status}"
        ).add_to(m)
    
    return m

def create_analysis_charts(aggregated, woreda_data, period_info, agg_level, agg_thresh):
    """Create analysis charts using Plotly"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Violence-Affected Areas (Ranked by Woreda Share)',
            'Population vs Violence Deaths',
            'Distribution of Violence Levels',
            'Woreda Classification Breakdown'
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "domain"}, {"type": "xy"}]]
    )
    
    # Chart 1: Horizontal bar chart of units with violence
    aggregated_nonzero = aggregated[aggregated['share_woredas_affected'] > 0].sort_values('share_woredas_affected', ascending=True)
    
    if len(aggregated_nonzero) > 0:
        name_col = f'{agg_level}_EN'
        colors = ['#d73027' if above else '#fd8d3c' for above in aggregated_nonzero['above_threshold']]
        
        fig.add_trace(
            go.Bar(
                y=[name[:20] + '...' if len(name) > 20 else name for name in aggregated_nonzero[name_col]],
                x=aggregated_nonzero['share_woredas_affected'],
                orientation='h',
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_vline(x=agg_thresh, line_dash="dash", line_color="red", row=1, col=1)
    
    # Chart 2: Population vs Deaths scatter
    if len(aggregated) > 0:
        scatter_colors = ['#d73027' if above else '#2c7fb8' for above in aggregated['above_threshold']]
        
        fig.add_trace(
            go.Scatter(
                x=aggregated['pop_count']/1000,
                y=aggregated['ACLED_BRD_total'],
                mode='markers',
                marker=dict(color=scatter_colors, size=8),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Chart 3: Distribution of violence levels
    if len(aggregated) > 0:
        aggregated_copy = aggregated.copy()
        aggregated_copy['violence_level'] = 'No Violence'
        aggregated_copy.loc[aggregated_copy['share_woredas_affected'] > 0, 'violence_level'] = 'Some Violence'
        aggregated_copy.loc[aggregated_copy['above_threshold'], 'violence_level'] = 'High Violence'
        
        level_counts = aggregated_copy['violence_level'].value_counts()
        colors_pie = ['#2c7fb8', '#fd8d3c', '#d73027']
        
        fig.add_trace(
            go.Pie(
                labels=level_counts.index,
                values=level_counts.values,
                marker_colors=colors_pie,
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Chart 4: Woreda classification breakdown
    if len(woreda_data) > 0:
        no_violence = len(woreda_data[woreda_data['ACLED_BRD_total'] == 0])
        below_threshold = len(woreda_data[(woreda_data['ACLED_BRD_total'] > 0) & (~woreda_data['violence_affected'])])
        violence_affected = len(woreda_data[woreda_data['violence_affected']])
        
        categories = ['No Violence', 'Below Threshold', 'Violence Affected']
        values = [no_violence, below_threshold, violence_affected]
        colors = ['#2c7fb8', '#fd8d3c', '#d73027']
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text=f'Supporting Analysis - {period_info["label"]}',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Share of Woredas Affected", row=1, col=1)
    fig.update_xaxes(title_text="Population (thousands)", row=1, col=2)
    fig.update_xaxes(title_text="Category", row=2, col=2)
    
    fig.update_yaxes(title_text="Administrative Unit", row=1, col=1)
    fig.update_yaxes(title_text="Total Deaths", row=1, col=2)
    fig.update_yaxes(title_text="Number of Woredas", row=2, col=2)
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇪🇹 Ethiopia Violence Analysis Dashboard</h1>
        <p>Interactive analysis with 12-month periods and detailed woreda mapping</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        periods = generate_12_month_periods()
        pop_data = load_population_data()
        conflict_data = load_conflict_data()
        boundaries = load_admin_boundaries()
    
    if pop_data.empty or conflict_data.empty:
        st.error("Failed to load required data. Please check your data files.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Violence Classification
    st.sidebar.subheader("📊 Violence Classification")
    rate_thresh = st.sidebar.slider(
        "Death Rate Threshold (per 100k)",
        min_value=0.5, max_value=20.0, value=4.0, step=0.5
    )
    abs_thresh = st.sidebar.slider(
        "Min Deaths Threshold",
        min_value=1, max_value=100, value=20, step=1
    )
    agg_thresh = st.sidebar.slider(
        "Aggregation Threshold",
        min_value=0.05, max_value=0.5, value=0.2, step=0.05
    )
    
    # Analysis Settings
    st.sidebar.subheader("🗺️ Analysis Settings")
    period_labels = [p['label'] for p in periods]
    period_idx = st.sidebar.selectbox(
        "Analysis Period",
        options=range(len(periods)),
        format_func=lambda x: periods[x]['label'],
        index=len(periods) - 10
    )
    
    agg_level = st.sidebar.selectbox(
        "Administrative Level",
        options=['ADM1', 'ADM2'],
        format_func=lambda x: 'Admin 1 (Regions)' if x == 'ADM1' else 'Admin 2 (Zones)',
        index=1
    )
    
    map_var = st.sidebar.selectbox(
        "Map Variable",
        options=['share_woredas', 'share_population'],
        format_func=lambda x: 'Share of Woredas Affected' if x == 'share_woredas' else 'Share of Population Affected',
        index=0
    )
    
    # Get selected period info
    period_info = periods[period_idx]
    
    # Status information
    st.markdown(f"""
    <div class="status-info">
        <strong>📊 Current Analysis Configuration</strong><br>
        <strong>Period:</strong> {period_info['label']} ({period_info['type'].replace('_', ' ').title()}) | 
        <strong>Level:</strong> {agg_level} | 
        <strong>Map Variable:</strong> {map_var}<br>
        <strong>Woreda Classification:</strong> Death rate >{rate_thresh:.1f}/100k AND >{abs_thresh} deaths<br>
        <strong>Unit Threshold:</strong> >{agg_thresh:.1%} of woredas affected → Marked as high-violence area
    </div>
    """, unsafe_allow_html=True)
    
    # Process data
    with st.spinner("Processing data..."):
        aggregated, woreda_data = classify_and_aggregate_data(
            pop_data, conflict_data, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level
        )
    
    # Metrics
    if len(aggregated) > 0:
        total_units = len(aggregated)
        above_threshold_count = aggregated['above_threshold'].sum()
        total_woredas = aggregated['total_woredas'].sum()
        affected_woredas = aggregated['violence_affected'].sum()
        total_population = aggregated['pop_count'].sum()
        affected_population = aggregated['affected_population'].sum()
        total_deaths = aggregated['ACLED_BRD_total'].sum()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 High Violence Units</h4>
                <div style="font-size: 24px; font-weight: bold;">{above_threshold_count}</div>
                <div>out of {total_units} ({above_threshold_count/total_units*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏘️ Affected Woredas</h4>
                <div style="font-size: 24px; font-weight: bold;">{affected_woredas}</div>
                <div>out of {total_woredas} ({affected_woredas/total_woredas*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>👥 Affected Population</h4>
                <div style="font-size: 24px; font-weight: bold;">{affected_population:,.0f}</div>
                <div>out of {total_population:,.0f} ({affected_population/total_population*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💀 Total Deaths</h4>
                <div style="font-size: 24px; font-weight: bold;">{total_deaths:,}</div>
                <div>in {period_info['label']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Maps
    st.header("🗺️ Interactive Violence Maps")
    
    tab1, tab2 = st.tabs(["📍 Administrative Units", "🏘️ Woreda Classification"])
    
    with tab1:
        st.subheader("Administrative Units Analysis")
        if len(aggregated) > 0:
            admin_map = create_admin_map(
                aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh
            )
            st_folium(admin_map, width=700, height=500)
        else:
            st.warning("No administrative data available for the selected period.")
    
    with tab2:
        st.subheader("Woreda Classification")
        if len(woreda_data) > 0:
            woreda_map = create_woreda_map(
                woreda_data, boundaries, period_info, rate_thresh, abs_thresh
            )
            st_folium(woreda_map, width=700, height=500)
        else:
            st.warning("No woreda data available for the selected period.")
    
    # Analysis Charts
    st.header("📈 Supporting Analysis")
    
    if len(aggregated) > 0 and len(woreda_data) > 0:
        analysis_fig = create_analysis_charts(
            aggregated, woreda_data, period_info, agg_level, agg_thresh
        )
        st.plotly_chart(analysis_fig, use_container_width=True)
    else:
        st.warning("No data available for analysis charts.")
    
    # Data Export
    st.header("📥 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Aggregated Data"):
            if len(aggregated) > 0:
                csv = aggregated.to_csv(index=False)
                st.download_button(
                    label="Download Aggregated Data CSV",
                    data=csv,
                    file_name=f"ethiopia_aggregated_{period_info['label'].replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No aggregated data to export.")
    
    with col2:
        if st.button("🏘️ Export Woreda Data"):
            if len(woreda_data) > 0:
                csv = woreda_data.to_csv(index=False)
                st.download_button(
                    label="Download Woreda Data CSV",
                    data=csv,
                    file_name=f"ethiopia_woredas_{period_info['label'].replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No woreda data to export.")

# Main app navigation - simplified to single page
def app():
    """Main application - single page dashboard"""
    main()

if __name__ == "__main__":
    app()

# Requirements.txt content (add this as a separate file)
"""
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
geopandas>=0.13.0
rasterio>=1.3.0
folium>=0.14.0
streamlit-folium>=0.13.0
plotly>=5.0.0
pyproj>=3.4.0
fiona>=1.8.0
"""

# Alternative requirements.txt for conda users:
"""
# If using conda, create environment first:
# conda create -n ethiopia-dashboard python=3.11
# conda activate ethiopia-dashboard
# conda install -c conda-forge geopandas rasterio folium plotly streamlit pyproj fiona
# pip install streamlit-folium
"""

# Instructions for deployment:
"""
INSTALLATION TROUBLESHOOTING:

1. The main issue is with pyproj/geopandas installation. Try these solutions:

SOLUTION 1 - Use conda (RECOMMENDED):
conda create -n ethiopia-dashboard python=3.11
conda activate ethiopia-dashboard
conda install -c conda-forge geopandas rasterio folium plotly streamlit pyproj fiona
pip install streamlit-folium

SOLUTION 2 - Fix pip installation:
pip uninstall geopandas pyproj fiona rasterio
pip install --upgrade pip setuptools wheel
pip install pyproj --no-binary pyproj
pip install fiona --no-binary fiona  
pip install geopandas rasterio folium plotly streamlit
pip install streamlit-folium

SOLUTION 3 - Windows-specific:
Download and install Microsoft Visual C++ Redistributable
pip install pipwin
pipwin install gdal fiona pyproj shapely
pip install geopandas rasterio folium plotly streamlit streamlit-folium

2. Create requirements.txt file:
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
geopandas>=0.13.0
rasterio>=1.3.0
folium>=0.14.0
streamlit-folium>=0.13.0
plotly>=5.0.0
pyproj>=3.4.0
fiona>=1.8.0

3. Ensure your data directory structure matches:
   data/
   ├── processed/
   │   └── intersection_result_acled.csv
   ├── eth_ppp_2020.tif
   └── eth_adm_csa_bofedb_2021_shp/
       ├── eth_admbnda_adm1_csa_bofedb_2021.shp (+ .dbf, .shx, .prj files)
       ├── eth_admbnda_adm2_csa_bofedb_2021.shp (+ .dbf, .shx, .prj files)
       └── eth_admbnda_adm3_csa_bofedb_2021.shp (+ .dbf, .shx, .prj files)

4. Run locally: streamlit run app.py

DEPLOYMENT OPTIONS:
- Streamlit Cloud (recommended for simple deployment)
- Heroku (requires Aptfile for GDAL dependencies)
- Docker (for consistent environment)

Key features:
- Interactive maps with administrative and woreda views
- 12-month period analysis (calendar and mid-year)
- Comprehensive metrics and visualizations
- Data export capabilities
- Responsive design with proper error handling
- Improved error handling for missing dependencies
"""