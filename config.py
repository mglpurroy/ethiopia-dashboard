# Configuration file for Ethiopia Violence Analysis Dashboard
# Optimized settings for better performance

import os
from pathlib import Path

# Performance settings
PERFORMANCE_CONFIG = {
    # Data processing
    'CHUNK_SIZE': int(os.getenv('CHUNK_SIZE', 10000)),  # For reading large CSV files
    'BATCH_SIZE': int(os.getenv('BATCH_SIZE', 50)),     # For processing geospatial data
    'MAX_WORKERS': int(os.getenv('MAX_WORKERS', 4)),    # For parallel processing
    
    # Caching settings
    'CACHE_TTL': int(os.getenv('CACHE_TTL', 3600)),     # Cache time-to-live in seconds
    'ENABLE_FILE_CACHE': os.getenv('ENABLE_FILE_CACHE', 'true').lower() == 'true',
    'CACHE_DIR': os.getenv('CACHE_DIR', 'cache'),
    
    # Memory management
    'GC_FREQUENCY': int(os.getenv('GC_FREQUENCY', 200)),  # Garbage collection frequency
    'MAX_MEMORY_MB': int(os.getenv('MAX_MEMORY_MB', 2048)),  # Maximum memory usage
    
    # Map rendering
    'MAP_PREFER_CANVAS': os.getenv('MAP_PREFER_CANVAS', 'true').lower() == 'true',
    'MAP_SIMPLIFY_TOLERANCE': float(os.getenv('MAP_SIMPLIFY_TOLERANCE', 0.001)),
    'MAX_MAP_FEATURES': int(os.getenv('MAX_MAP_FEATURES', 1000)),
    
    # UI settings
    'SHOW_PERFORMANCE_METRICS': os.getenv('SHOW_PERFORMANCE_METRICS', 'true').lower() == 'true',
    'ENABLE_PROGRESS_BARS': os.getenv('ENABLE_PROGRESS_BARS', 'true').lower() == 'true',
}

# Data paths
DATA_CONFIG = {
    'DATA_PATH': Path(os.getenv('DATA_PATH', 'data')),
    'PROCESSED_PATH': Path(os.getenv('PROCESSED_PATH', 'data/processed')),
    'POPULATION_RASTER': os.getenv('POPULATION_RASTER', 'data/eth_ppp_2020.tif'),
    'ADMIN_SHAPEFILES': {
        1: os.getenv('ADMIN1_SHAPEFILE', 'data/eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm1_csa_bofedb_2021.shp'),
        2: os.getenv('ADMIN2_SHAPEFILE', 'data/eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm2_csa_bofedb_2021.shp'),
        3: os.getenv('ADMIN3_SHAPEFILE', 'data/eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm3_csa_bofedb_2021.shp'),
    },
    'CONFLICT_DATA': os.getenv('CONFLICT_DATA', 'data/processed/intersection_result_acled.csv'),
}

# Analysis settings
ANALYSIS_CONFIG = {
    'START_YEAR': int(os.getenv('START_YEAR', 2009)),
    'END_YEAR': int(os.getenv('END_YEAR', 2025)),
    'DEFAULT_RATE_THRESH': float(os.getenv('DEFAULT_RATE_THRESH', 4.0)),
    'DEFAULT_ABS_THRESH': int(os.getenv('DEFAULT_ABS_THRESH', 20)),
    'DEFAULT_AGG_THRESH': float(os.getenv('DEFAULT_AGG_THRESH', 0.2)),
}

# Streamlit page config
PAGE_CONFIG = {
    'page_title': "Ethiopia Violence Analysis Dashboard",
    'page_icon': "🇪🇹",
    'layout': "wide",
    'initial_sidebar_state': "expanded",
    'menu_items': {
        'Get Help': 'https://github.com/your-repo/issues',
        'Report a bug': 'https://github.com/your-repo/issues/new',
        'About': """
        # Ethiopia Violence Analysis Dashboard
        
        An optimized interactive dashboard for analyzing violence patterns in Ethiopia.
        
        **Features:**
        - Real-time data processing with caching
        - Interactive maps with performance optimization
        - Comprehensive statistical analysis
        - Export capabilities
        - Performance monitoring
        
        **Version:** 2.0 (Optimized)
        """
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'enable_performance_logging': os.getenv('ENABLE_PERFORMANCE_LOGGING', 'true').lower() == 'true',
}