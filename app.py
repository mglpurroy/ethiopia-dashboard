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
import gc
from functools import lru_cache
import pickle
import hashlib
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ethiopia Violence Analysis Dashboard",
    page_icon="🇪🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance monitoring
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}

def log_performance(func_name, duration):
    """Log performance metrics for monitoring"""
    if func_name not in st.session_state.performance_metrics:
        st.session_state.performance_metrics[func_name] = []
    st.session_state.performance_metrics[func_name].append(duration)

st.info("🚧 **BETA VERSION** - This dashboard is currently in beta testing")

# Custom CSS - optimized for better performance
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
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
    .performance-info {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.8rem;
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
    /* Optimize map rendering */
    .element-container iframe {
        width: 100% !important;
    }
    /* Loading spinner optimization */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Data paths - adjust these for your deployment
DATA_PATH = Path("data/")
PROCESSED_PATH = DATA_PATH / "processed"
CACHE_PATH = Path("cache/")
CACHE_PATH.mkdir(exist_ok=True)

POPULATION_RASTER = DATA_PATH / "eth_ppp_2020.tif"
ADMIN_SHAPEFILES = {
    1: DATA_PATH / "eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm1_csa_bofedb_2021.shp",
    2: DATA_PATH / "eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm2_csa_bofedb_2021.shp",
    3: DATA_PATH / "eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm3_csa_bofedb_2021.shp"
}

START_YEAR = 2009
END_YEAR = 2025

def get_cache_key(*args):
    """Generate cache key from arguments"""
    return hashlib.md5(str(args).encode()).hexdigest()

def save_to_cache(key, data):
    """Save data to cache file"""
    try:
        cache_file = CACHE_PATH / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"Cache save failed: {e}")

def load_from_cache(key):
    """Load data from cache file"""
    try:
        cache_file = CACHE_PATH / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Cache load failed: {e}")
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_12_month_periods():
    """Generate 12-month periods every 6 months - optimized"""
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

@st.cache_data(ttl=3600, show_spinner=False)
def load_population_data():
    """Load and cache population data with optimized processing"""
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("population_data", "v2")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_population_data", time.time() - start_time)
        st.success(f"✅ Loaded cached population data for {len(cached_data)} woredas")
        return cached_data
    
    try:
        # Check if shapefile exists
        if not ADMIN_SHAPEFILES[3].exists():
            st.error(f"Shapefile not found: {ADMIN_SHAPEFILES[3]}")
            return pd.DataFrame()
        
        # Load shapefile with optimized settings
        admin3_gdf = gpd.read_file(ADMIN_SHAPEFILES[3])
        
        # Optimize geometry for faster processing
        admin3_gdf = admin3_gdf.to_crs('EPSG:4326')  # Ensure consistent CRS
        
        # Check if raster exists and process if available
        if POPULATION_RASTER.exists():
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
                    
                    # Process in batches for better memory management
                    batch_size = 50
                    for batch_start in range(0, total_rows, batch_size):
                        batch_end = min(batch_start + batch_size, total_rows)
                        batch_gdf = admin3_gdf.iloc[batch_start:batch_end]
                        
                        for idx, row in batch_gdf.iterrows():
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
                            except Exception:
                                # Skip problematic rows but continue processing
                                continue
                        
                        # Update progress
                        progress_bar.progress(batch_end / total_rows)
                        
                        # Force garbage collection to manage memory
                        if batch_start % (batch_size * 4) == 0:
                            gc.collect()
                    
                    progress_bar.empty()
                    
                    if not population_data:
                        st.error("No population data could be extracted from raster.")
                        return pd.DataFrame()
                    
                    result_df = pd.DataFrame(population_data)
                    
                    # Cache the result
                    save_to_cache(cache_key, result_df)
                    
                    log_performance("load_population_data", time.time() - start_time)
                    st.success(f"✅ Loaded population data for {len(population_data)} woredas")
                    return result_df
                    
            except Exception as raster_error:
                st.warning(f"Error processing population raster: {raster_error}")
                st.info("Using simplified population estimates...")
        
        # Fallback: Create simplified population data
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
        
        result_df = pd.DataFrame(simplified_data)
        
        # Cache the result
        save_to_cache(cache_key, result_df)
        
        log_performance("load_population_data", time.time() - start_time)
        st.info(f"✅ Using simplified population estimates for {len(simplified_data)} woredas")
        return result_df
        
    except Exception as e:
        st.error(f"Error loading population data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def create_admin_levels(pop_data):
    """Create admin level aggregations from population data - optimized"""
    import time
    start_time = time.time()
    
    if pop_data.empty:
        return {'admin1': pd.DataFrame(), 'admin2': pd.DataFrame(), 'admin3': pop_data}
    
    # Use vectorized operations for better performance
    admin2_agg = pop_data.groupby(['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'ADM0_PCODE'], as_index=False).agg({
        'pop_count': 'sum',
        'pop_count_millions': 'sum'
    })
    
    admin1_agg = pop_data.groupby(['ADM1_PCODE', 'ADM1_EN', 'ADM0_PCODE'], as_index=False).agg({
        'pop_count': 'sum',
        'pop_count_millions': 'sum'
    })
    
    log_performance("create_admin_levels", time.time() - start_time)
    
    return {
        'admin3': pop_data,
        'admin2': admin2_agg,
        'admin1': admin1_agg
    }

@st.cache_data(ttl=3600, show_spinner=False)
def load_conflict_data():
    """Load and cache conflict data with optimized processing"""
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("conflict_data", "v2")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_conflict_data", time.time() - start_time)
        st.success(f"✅ Loaded cached conflict data: {len(cached_data)} records")
        return cached_data
    
    try:
        conflict_file = PROCESSED_PATH / "intersection_result_acled.csv"
        if not conflict_file.exists():
            st.error(f"Conflict data not found: {conflict_file}")
            return pd.DataFrame()
        
        # Load data with optimized dtypes
        dtypes = {
            'GID_0': 'category',
            'year': 'int16',
            'month': 'int8',
            'ACLED_BRD_state': 'float32',
            'ACLED_BRD_nonstate': 'float32',
            'ACLED_BRD_total': 'float32'
        }
        
        # Load in chunks for memory efficiency
        chunk_size = 10000
        chunks = []
        
        for chunk in pd.read_csv(conflict_file, chunksize=chunk_size, dtype=dtypes):
            ethiopia_chunk = chunk[chunk['GID_0'] == 'ETH'].copy()
            if not ethiopia_chunk.empty:
                chunks.append(ethiopia_chunk)
        
        if not chunks:
            st.warning("No Ethiopia data found in conflict file")
            return pd.DataFrame()
        
        # Combine chunks
        ethiopia_acled = pd.concat(chunks, ignore_index=True)
        
        # Optimize data processing
        if 'GID_3' in ethiopia_acled.columns:
            # Use efficient aggregation
            conflict_mapped = ethiopia_acled.groupby(['year', 'month', 'GID_3'], as_index=False).agg({
                'ACLED_BRD_state': 'sum',
                'ACLED_BRD_nonstate': 'sum',
                'ACLED_BRD_total': 'sum'
            })
            
            # Rename GID_3 to ADM3_PCODE for consistency
            conflict_mapped = conflict_mapped.rename(columns={'GID_3': 'ADM3_PCODE'})
            
            # Cache the result
            save_to_cache(cache_key, conflict_mapped)
            
            log_performance("load_conflict_data", time.time() - start_time)
            st.success(f"✅ Loaded conflict data: {len(conflict_mapped)} records")
            return conflict_mapped
        else:
            st.warning("GID_3 column not found in conflict data")
            return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading conflict data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_admin_boundaries():
    """Load administrative boundaries with optimized processing"""
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("admin_boundaries", "v2")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_admin_boundaries", time.time() - start_time)
        st.success("✅ Loaded cached administrative boundaries")
        return cached_data
    
    boundaries = {}
    for level in [1, 2, 3]:
        try:
            if not ADMIN_SHAPEFILES[level].exists():
                st.warning(f"Shapefile not found: {ADMIN_SHAPEFILES[level]}")
                boundaries[level] = gpd.GeoDataFrame()
                continue
            
            # Load with optimized settings
            gdf = gpd.read_file(ADMIN_SHAPEFILES[level])
            
            # Simplify geometries for better performance (reduce precision)
            if level <= 2:  # Only simplify for higher level boundaries
                gdf.geometry = gdf.geometry.simplify(tolerance=0.001, preserve_topology=True)
            
            # Ensure consistent CRS
            gdf = gdf.to_crs('EPSG:4326')
            
            boundaries[level] = gdf
            st.success(f"✅ Loaded Admin {level} boundaries: {len(gdf)} features")
            
        except Exception as e:
            st.warning(f"Error loading admin {level} boundaries: {str(e)}")
            boundaries[level] = gpd.GeoDataFrame()
    
    # Cache the result
    save_to_cache(cache_key, boundaries)
    
    log_performance("load_admin_boundaries", time.time() - start_time)
    return boundaries

@lru_cache(maxsize=128)
def filter_data_by_period(data_hash, period_info_str):
    """Filter data based on 12-month period - cached version"""
    # This is a placeholder - actual implementation would need to handle the data properly
    # The caching is done at a higher level in the processing pipeline
    pass

def filter_data_by_period_impl(data, period_info):
    """Filter data based on 12-month period - optimized implementation"""
    if len(data) == 0:
        return data
    
    start_year = period_info['start_year']
    end_year = period_info['end_year']
    start_month = period_info['start_month']
    end_month = period_info['end_month']
    
    if period_info['type'] == 'calendar':
        # Calendar year: Jan-Dec - use vectorized operations
        mask = (data['year'] == start_year) & (data['month'] >= start_month) & (data['month'] <= end_month)
        return data[mask]
    else:
        # Mid-year: Jul-Jun - use vectorized operations
        mask = ((data['year'] == start_year) & (data['month'] >= start_month)) | \
               ((data['year'] == end_year) & (data['month'] <= end_month))
        return data[mask]

def classify_and_aggregate_data(pop_data, admin_data, conflict_data, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level):
    """Classify woredas and aggregate to selected administrative level - optimized"""
    import time
    start_time = time.time()
    
    # Filter conflict data for selected period using optimized function
    period_conflict = filter_data_by_period_impl(conflict_data, period_info)
    
    # Aggregate conflict data by woreda for the period using vectorized operations
    if len(period_conflict) > 0:
        conflict_agg = period_conflict.groupby('ADM3_PCODE', as_index=False).agg({
            'ACLED_BRD_state': 'sum',
            'ACLED_BRD_nonstate': 'sum',
            'ACLED_BRD_total': 'sum'
        })
    else:
        conflict_agg = pd.DataFrame()
    
    # Merge with population data using optimized merge
    if len(conflict_agg) > 0:
        merged = pd.merge(pop_data, conflict_agg, on='ADM3_PCODE', how='left')
        # Use vectorized fillna
        conflict_cols = ['ACLED_BRD_state', 'ACLED_BRD_nonstate', 'ACLED_BRD_total']
        merged[conflict_cols] = merged[conflict_cols].fillna(0)
    else:
        merged = pop_data.copy()
        for col in ['ACLED_BRD_state', 'ACLED_BRD_nonstate', 'ACLED_BRD_total']:
            merged[col] = 0
    
    # Calculate death rates using vectorized operations
    merged['acled_total_death_rate'] = (merged['ACLED_BRD_total'] / (merged['pop_count_millions'] * 1e6)) * 1e5
    
    # Classify woredas using vectorized operations
    merged['violence_affected'] = (
        (merged['acled_total_death_rate'] > rate_thresh) & 
        (merged['ACLED_BRD_total'] > abs_thresh)
    )
    
    # Aggregate to selected level using optimized groupby
    if agg_level == 'ADM1':
        group_cols = ['ADM1_PCODE', 'ADM1_EN']
    else:  # ADM2
        group_cols = ['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN']
    
    aggregated = merged.groupby(group_cols, as_index=False).agg({
        'pop_count': 'sum',
        'violence_affected': 'sum',
        'ADM3_PCODE': 'count',
        'ACLED_BRD_total': 'sum'
    })
    
    aggregated.rename(columns={'ADM3_PCODE': 'total_woredas'}, inplace=True)
    
    # Calculate shares using vectorized operations
    aggregated['share_woredas_affected'] = aggregated['violence_affected'] / aggregated['total_woredas']
    
    # Calculate population share using optimized operations
    affected_pop = merged[merged['violence_affected']].groupby(group_cols[0], as_index=False)['pop_count'].sum()
    affected_pop.rename(columns={'pop_count': 'affected_population'}, inplace=True)
    aggregated = pd.merge(aggregated, affected_pop, on=group_cols[0], how='left')
    aggregated['affected_population'] = aggregated['affected_population'].fillna(0)
    aggregated['share_population_affected'] = aggregated['affected_population'] / aggregated['pop_count']
    
    # Mark units above threshold using vectorized operations
    aggregated['above_threshold'] = aggregated['share_woredas_affected'] > agg_thresh
    
    log_performance("classify_and_aggregate_data", time.time() - start_time)
    
    return aggregated, merged

def create_optimized_map(gdf, value_col, value_label, threshold, is_woreda=False):
    """Create optimized map with reduced complexity"""
    import time
    start_time = time.time()
    
    # Create map with optimized settings
    m = folium.Map(
        location=[9.15, 40.49], 
        zoom_start=6, 
        tiles='OpenStreetMap',
        prefer_canvas=True  # Use canvas for better performance
    )
    
    # Reduce the number of features for better performance
    if len(gdf) > 1000:
        st.warning("Large dataset detected. Map rendering may be slow.")
    
    # Add features with optimized styling
    for idx, (_, row) in enumerate(gdf.iterrows()):
        if idx % 100 == 0:  # Progress indicator for large datasets
            pass
        
        value = row.get(value_col, 0)
        
        # Determine color and status
        if is_woreda:
            if row.get('violence_affected', False):
                color = '#d73027'
                status = "VIOLENCE AFFECTED"
                opacity = 0.8
            elif row.get('ACLED_BRD_total', 0) > 0:
                color = '#fd8d3c'
                status = "Below Threshold"
                opacity = 0.6
            else:
                color = '#2c7fb8'
                status = "No Violence"
                opacity = 0.3
        else:
            if value > threshold:
                color = '#d73027'
                status = "HIGH VIOLENCE"
                opacity = 0.8
            elif value > 0:
                color = '#fd8d3c'
                status = "Some Violence"
                opacity = 0.7
            else:
                color = '#2c7fb8'
                status = "Low/No Violence"
                opacity = 0.4
        
        # Simplified popup content
        popup_content = f"""
        <div style="width: 250px; font-family: Arial, sans-serif;">
            <h4 style="color: {color}; margin: 0;">{row.get('ADM3_EN' if is_woreda else f'ADM{1 if "ADM1" in str(row) else 2}_EN', 'Unknown')}</h4>
            <div style="background: {color}; color: white; padding: 3px; border-radius: 2px; text-align: center; margin: 5px 0;">
                <strong>{status}</strong>
            </div>
            <p><strong>{value_label}:</strong> {value:.1%}</p>
        </div>
        """
        
        # Add to map with optimized settings
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5 if is_woreda else 1,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=280),
            tooltip=f"{row.get('ADM3_EN' if is_woreda else f'ADM{1 if "ADM1" in str(row) else 2}_EN', 'Unknown')}: {status}"
        ).add_to(m)
    
    log_performance("create_optimized_map", time.time() - start_time)
    return m

def create_admin_map(aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh):
    """Create administrative units map with optimized performance"""
    import time
    start_time = time.time()
    
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
    
    if gdf.empty:
        st.error(f"No boundary data available for {agg_level}")
        return None
    
    # Merge data with boundaries using optimized merge
    merge_cols = [pcode_col, value_col, 'above_threshold', 'violence_affected', 'total_woredas', 'pop_count', 'ACLED_BRD_total']
    merged_gdf = gdf.merge(aggregated[merge_cols], on=pcode_col, how='left')
    
    # Use vectorized fillna
    fill_values = {
        value_col: 0, 
        'above_threshold': False, 
        'violence_affected': 0, 
        'total_woredas': 0,
        'pop_count': 0,
        'ACLED_BRD_total': 0
    }
    merged_gdf = merged_gdf.fillna(fill_values)
    
    # Create map with optimized settings
    m = folium.Map(
        location=[9.15, 40.49], 
        zoom_start=6, 
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Pre-calculate colors and status for better performance
    def get_color_status(value):
        if value > agg_thresh:
            return '#d73027', 0.8, "HIGH VIOLENCE"
        elif value > 0:
            return '#fd8d3c', 0.7, "Some Violence"
        else:
            return '#2c7fb8', 0.4, "Low/No Violence"
    
    # Add choropleth layer with optimized rendering
    for _, row in merged_gdf.iterrows():
        value = row[value_col]
        color, opacity, status = get_color_status(value)
        
        # Simplified popup content for better performance
        popup_content = f"""
        <div style="width: 280px; font-family: Arial, sans-serif;">
            <h4 style="color: {color}; margin: 0;">{row.get(name_col, 'Unknown')}</h4>
            <div style="background: {color}; color: white; padding: 3px; border-radius: 2px; text-align: center; margin: 5px 0;">
                <strong>{status}</strong>
            </div>
            <p><strong>{value_label}:</strong> {value:.1%}</p>
            <p><strong>Affected Woredas:</strong> {row['violence_affected']}/{row['total_woredas']}</p>
            <p><strong>Total Deaths:</strong> {row['ACLED_BRD_total']:,.0f}</p>
        </div>
        """
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.8,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{row.get(name_col, 'Unknown')}: {value:.1%}"
        ).add_to(m)
    
    # Simplified legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 250px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                border-radius: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #333;">{value_label}</h4>
    <div style="margin-bottom: 6px;">
        <div style="margin: 2px 0;"><span style="background:#d73027; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">HIGH</span> >{agg_thresh:.1%}</div>
        <div style="margin: 2px 0;"><span style="background:#fd8d3c; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">SOME</span> >0%</div>
        <div style="margin: 2px 0;"><span style="background:#2c7fb8; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">LOW</span> 0%</div>
    </div>
    <div style="font-size:9px; color:#666;">
        <strong>Period:</strong> {period_info['label']}<br>
        <strong>Criteria:</strong> >{rate_thresh:.1f}/100k & >{abs_thresh} deaths
    </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    log_performance("create_admin_map", time.time() - start_time)
    return m

def create_woreda_map(woreda_data, boundaries, period_info, rate_thresh, abs_thresh):
    """Create woreda classification map with optimized performance"""
    import time
    start_time = time.time()
    
    # Get woreda boundaries
    woreda_gdf = boundaries[3]
    
    if woreda_gdf.empty:
        st.error("No woreda boundary data available")
        return None
    
    # Merge with classification data using optimized merge
    merge_cols = ['ADM3_PCODE', 'violence_affected', 'ACLED_BRD_total', 'acled_total_death_rate']
    merged_woreda = woreda_gdf.merge(woreda_data[merge_cols], on='ADM3_PCODE', how='left')
    
    # Use vectorized fillna
    fill_values = {
        'violence_affected': False, 
        'ACLED_BRD_total': 0,
        'acled_total_death_rate': 0
    }
    merged_woreda = merged_woreda.fillna(fill_values)
    
    # Create map with optimized settings
    m = folium.Map(
        location=[9.15, 40.49], 
        zoom_start=6, 
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Pre-calculate statistics for legend
    total_woredas = len(woreda_data)
    affected_woredas = sum(woreda_data['violence_affected'])
    affected_percentage = (affected_woredas / total_woredas * 100) if total_woredas > 0 else 0
    
    # Pre-calculate colors and status for better performance
    def get_woreda_color_status(row):
        if row['violence_affected']:
            return '#d73027', 0.8, "VIOLENCE AFFECTED"
        elif row['ACLED_BRD_total'] > 0:
            return '#fd8d3c', 0.6, "Below Threshold"
        else:
            return '#2c7fb8', 0.3, "No Violence"
    
    # Add woreda layer with optimized rendering
    for _, row in merged_woreda.iterrows():
        color, opacity, status = get_woreda_color_status(row)
        
        # Simplified popup content for better performance
        popup_content = f"""
        <div style="width: 250px; font-family: Arial, sans-serif;">
            <h4 style="color: {color}; margin: 0;">{row.get('ADM3_EN', 'Unknown')}</h4>
            <div style="background: {color}; color: white; padding: 3px; border-radius: 2px; text-align: center; margin: 5px 0;">
                <strong>{status}</strong>
            </div>
            <p><strong>Zone:</strong> {row.get('ADM2_EN', 'Unknown')}</p>
            <p><strong>Deaths:</strong> {row['ACLED_BRD_total']:,.0f}</p>
            <p><strong>Rate:</strong> {row['acled_total_death_rate']:.1f}/100k</p>
        </div>
        """
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.3,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=270),
            tooltip=f"{row.get('ADM3_EN', 'Unknown')}: {status}"
        ).add_to(m)
    
    # Simplified legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 240px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                border-radius: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #333;">Woreda Classification</h4>
    <div style="margin-bottom: 6px;">
        <div style="margin: 2px 0;"><span style="background:#d73027; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">AFFECTED</span> Violence Affected</div>
        <div style="margin: 2px 0;"><span style="background:#fd8d3c; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">BELOW</span> Below Threshold</div>
        <div style="margin: 2px 0;"><span style="background:#2c7fb8; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">NONE</span> No Violence</div>
    </div>
    <div style="font-size:9px; color:#666;">
        <strong>Period:</strong> {period_info['label']}<br>
        <strong>Criteria:</strong> >{rate_thresh:.1f}/100k & >{abs_thresh} deaths<br>
        <strong>Affected:</strong> {affected_woredas}/{total_woredas} ({affected_percentage:.1f}%)
    </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    log_performance("create_woreda_map", time.time() - start_time)
    return m

def create_analysis_charts(aggregated, woreda_data, period_info, agg_level, agg_thresh):
    """Create comprehensive analysis charts using Plotly"""
    
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
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Share: %{x:.1%}<extra></extra>'
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
                showlegend=False,
                hovertemplate='<b>Population:</b> %{x:.0f}k<br><b>Deaths:</b> %{y}<extra></extra>'
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
                showlegend=False,
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
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
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{y:.1%}<extra></extra>',
                text=[f'{v}<br>({v/len(woreda_data)*100:.1f}%)' for v in values],
                textposition='auto'
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
    """Main Streamlit application with optimized performance"""
    import time
    app_start_time = time.time()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇪🇹 Ethiopia Violence Analysis Dashboard</h1>
        <p>Interactive analysis with 12-month periods and comprehensive woreda mapping</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for data caching
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.periods = None
        st.session_state.pop_data = None
        st.session_state.admin_data = None
        st.session_state.conflict_data = None
        st.session_state.boundaries = None
    
    # Load data with progress indicators and session state caching
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            data_start_time = time.time()
            
            st.session_state.periods = generate_12_month_periods()
            st.session_state.pop_data = load_population_data()
            st.session_state.admin_data = create_admin_levels(st.session_state.pop_data)
            st.session_state.conflict_data = load_conflict_data()
            st.session_state.boundaries = load_admin_boundaries()
            
            st.session_state.data_loaded = True
            
            data_load_time = time.time() - data_start_time
            log_performance("data_loading", data_load_time)
            
            # Show performance info
            st.markdown(f"""
            <div class="performance-info">
                ⚡ Data loaded in {data_load_time:.2f} seconds
            </div>
            """, unsafe_allow_html=True)
    
    # Use cached data
    periods = st.session_state.periods
    pop_data = st.session_state.pop_data
    admin_data = st.session_state.admin_data
    conflict_data = st.session_state.conflict_data
    boundaries = st.session_state.boundaries
    
    if pop_data.empty:
        st.error("Failed to load population data. Please check your data files.")
        st.stop()
    
    if conflict_data.empty:
        st.warning("No conflict data available. Dashboard will show population data only.")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Violence Classification
    st.sidebar.subheader("📊 Violence Classification")
    rate_thresh = st.sidebar.slider(
        "Death Rate Threshold (per 100k)",
        min_value=0.5, max_value=20.0, value=4.0, step=0.5,
        help="Minimum death rate per 100,000 population to classify as violence-affected"
    )
    abs_thresh = st.sidebar.slider(
        "Min Deaths Threshold",
        min_value=1, max_value=100, value=20, step=1,
        help="Minimum absolute number of deaths to classify as violence-affected"
    )
    agg_thresh = st.sidebar.slider(
        "Aggregation Threshold",
        min_value=0.05, max_value=0.5, value=0.2, step=0.05,
        help="Minimum share of woredas affected to mark administrative unit as high-violence"
    )
    
    # Analysis Settings
    st.sidebar.subheader("🗺️ Analysis & Display Settings")
    period_labels = [p['label'] for p in periods]
    period_idx = st.sidebar.selectbox(
        "Analysis Period",
        options=range(len(periods)),
        format_func=lambda x: periods[x]['label'],
        index=len(periods) - 10,
        help="Select 12-month analysis period (calendar year or mid-year cycle)"
    )
    
    agg_level = st.sidebar.selectbox(
        "Administrative Level",
        options=['ADM1', 'ADM2'],
        format_func=lambda x: 'Admin 1 (Regions)' if x == 'ADM1' else 'Admin 2 (Zones)',
        index=1,
        help="Administrative level for aggregated analysis"
    )
    
    map_var = st.sidebar.selectbox(
        "Map Variable",
        options=['share_woredas', 'share_population'],
        format_func=lambda x: 'Share of Woredas Affected' if x == 'share_woredas' else 'Share of Population Affected',
        index=0,
        help="Variable to display on administrative units map"
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
    with st.spinner("Processing analysis..."):
        aggregated, woreda_data = classify_and_aggregate_data(
            pop_data, admin_data, conflict_data, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level
        )
    
    # Display metrics
    if len(aggregated) > 0:
        total_units = len(aggregated)
        above_threshold_count = aggregated['above_threshold'].sum()
        total_woredas = aggregated['total_woredas'].sum()
        affected_woredas = aggregated['violence_affected'].sum()
        total_population = aggregated['pop_count'].sum()
        affected_population = aggregated['affected_population'].sum()
        total_deaths = aggregated['ACLED_BRD_total'].sum()
        
        # Display metrics in a grid
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
                <h4>Total Deaths</h4>
                <div style="font-size: 24px; font-weight: bold;">{total_deaths:,}</div>
                <div>in {period_info['label']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Maps section
    st.header("🗺️ Interactive Violence Maps")
    st.markdown("**📍 Administrative Units**: Aggregated analysis by regions/zones | **🏘️ Woreda Classification**: Individual woreda violence classification")
    
    tab1, tab2 = st.tabs(["📍 Administrative Units", "🏘️ Woreda Classification"])
    
    with tab1:
        st.subheader(f"Administrative Units Analysis - {agg_level}")
        if len(aggregated) > 0:
            admin_map = create_admin_map(
                aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh
            )
            if admin_map:
                # Use full width for the map
                st_folium(admin_map, width=None, height=600, returned_objects=["last_object_clicked"])
            else:
                st.error("Could not create administrative map due to missing boundary data.")
        else:
            st.warning("No administrative data available for the selected period.")
    
    with tab2:
        st.subheader("Individual Woreda Classification")
        if len(woreda_data) > 0:
            woreda_map = create_woreda_map(
                woreda_data, boundaries, period_info, rate_thresh, abs_thresh
            )
            if woreda_map:
                # Use full width for the map
                st_folium(woreda_map, width=None, height=600, returned_objects=["last_object_clicked"])
            else:
                st.error("Could not create woreda map due to missing boundary data.")
        else:
            st.warning("No woreda data available for the selected period.")
    
    # Analysis Charts
    st.header("📈 Supporting Analysis & Insights")
    
    if len(aggregated) > 0 and len(woreda_data) > 0:
        analysis_fig = create_analysis_charts(
            aggregated, woreda_data, period_info, agg_level, agg_thresh
        )
        st.plotly_chart(analysis_fig, use_container_width=True)
        
        # Additional insights
        st.subheader("📋 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Violence Hotspots")
            if len(aggregated[aggregated['above_threshold']]) > 0:
                hotspots = aggregated[aggregated['above_threshold']].sort_values('share_woredas_affected', ascending=False)
                name_col = f'{agg_level}_EN'
                for idx, row in hotspots.head(5).iterrows():
                    st.markdown(f"**{row[name_col]}**: {row['share_woredas_affected']:.1%} woredas affected ({row['violence_affected']}/{row['total_woredas']})")
            else:
                st.markdown("No units above the violence threshold in this period.")
        
        with col2:
            st.markdown("### 📊 Statistical Summary")
            st.markdown(f"**Period**: {period_info['label']}")
            st.markdown(f"**Total Deaths**: {total_deaths:,}")
            st.markdown(f"**National Death Rate**: {(total_deaths/total_population)*1e5:.1f} per 100k")
            st.markdown(f"**Violence Coverage**: {affected_woredas/total_woredas*100:.1f}% of woredas")
            st.markdown(f"**Population Impact**: {affected_population/total_population*100:.1f}% of population")
    else:
        st.warning("No data available for analysis charts.")
    
    # Data Export Section
    st.header("📥 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Aggregated Data", use_container_width=True):
            if len(aggregated) > 0:
                csv = aggregated.to_csv(index=False)
                st.download_button(
                    label="Download Aggregated Data CSV",
                    data=csv,
                    file_name=f"ethiopia_aggregated_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No aggregated data to export.")
    
    with col2:
        if st.button("🏘️ Export Woreda Data", use_container_width=True):
            if len(woreda_data) > 0:
                csv = woreda_data.to_csv(index=False)
                st.download_button(
                    label="Download Woreda Data CSV",
                    data=csv,
                    file_name=f"ethiopia_woredas_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No woreda data to export.")
    
    with col3:
        if st.button("📈 Export Analysis Summary", use_container_width=True):
            if len(aggregated) > 0:
                summary_data = {
                    'period': [period_info['label']],
                    'period_type': [period_info['type']],
                    'total_units': [total_units],
                    'high_violence_units': [above_threshold_count],
                    'total_woredas': [total_woredas],
                    'affected_woredas': [affected_woredas],
                    'total_population': [total_population],
                    'affected_population': [affected_population],
                    'total_deaths': [total_deaths],
                    'national_death_rate': [(total_deaths/total_population)*1e5],
                    'analysis_timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"ethiopia_summary_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No summary data to export.")
    
    # Performance monitoring section (expandable)
    with st.expander("⚡ Performance Metrics", expanded=False):
        if st.session_state.performance_metrics:
            st.subheader("Function Performance")
            
            perf_data = []
            for func_name, times in st.session_state.performance_metrics.items():
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                call_count = len(times)
                
                perf_data.append({
                    'Function': func_name,
                    'Avg Time (s)': f"{avg_time:.3f}",
                    'Min Time (s)': f"{min_time:.3f}",
                    'Max Time (s)': f"{max_time:.3f}",
                    'Calls': call_count
                })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)
                
                # Performance tips
                st.markdown("""
                **Performance Tips:**
                - Data is cached in session state for faster subsequent loads
                - File-based caching reduces processing time for repeated operations
                - Vectorized operations improve pandas performance
                - Map rendering uses canvas mode for better performance
                - Simplified popups and legends reduce rendering overhead
                """)
        else:
            st.info("No performance data available yet.")
    
    # Add refresh button to clear cache
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Refresh Data Cache", use_container_width=True):
            # Clear session state
            for key in ['data_loaded', 'periods', 'pop_data', 'admin_data', 'conflict_data', 'boundaries']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear file cache
            import shutil
            if CACHE_PATH.exists():
                shutil.rmtree(CACHE_PATH)
                CACHE_PATH.mkdir(exist_ok=True)
            
            st.success("Cache cleared! Please refresh the page to reload data.")
            st.experimental_rerun()
    
    # Total app performance
    total_time = time.time() - app_start_time
    st.markdown(f"""
    <div class="performance-info">
        🏁 Total app processing time: {total_time:.2f} seconds
    </div>
    """, unsafe_allow_html=True)

# Main app navigation - simplified to single page
def app():
    """Main application - single page dashboard"""
    main()

if __name__ == "__main__":
    app()
