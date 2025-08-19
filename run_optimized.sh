#!/bin/bash
# Startup script for optimized Ethiopia Violence Analysis Dashboard

echo "🚀 Starting Ethiopia Violence Analysis Dashboard (Optimized Version)"
echo "=================================================="

# Set optimal environment variables for performance
export STREAMLIT_SERVER_MAXUPLOADSIZE=1000
export STREAMLIT_SERVER_MAXMESSAGESIZE=1000
export STREAMLIT_BROWSER_GATHERTUSAGESTAT=false

# Performance optimization settings
export CHUNK_SIZE=10000
export BATCH_SIZE=50
export CACHE_TTL=3600
export MAX_MEMORY_MB=2048
export MAP_PREFER_CANVAS=true
export MAP_SIMPLIFY_TOLERANCE=0.001
export MAX_MAP_FEATURES=1000
export SHOW_PERFORMANCE_METRICS=true
export ENABLE_PROGRESS_BARS=true
export ENABLE_FILE_CACHE=true

# Create cache directory if it doesn't exist
mkdir -p cache

# Check if virtual environment exists
if [ -d "eth_dashboard_env" ]; then
    echo "📦 Activating virtual environment..."
    source eth_dashboard_env/bin/activate
else
    echo "⚠️  Virtual environment not found. Using system Python."
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "import streamlit, pandas, geopandas, folium, plotly" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All required packages are available"
else
    echo "❌ Missing required packages. Please install from requirements.txt:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "⚠️  Data directory not found. Please ensure data files are in the 'data' directory."
fi

# Display performance settings
echo ""
echo "⚡ Performance Settings:"
echo "   - Chunk Size: $CHUNK_SIZE"
echo "   - Batch Size: $BATCH_SIZE"
echo "   - Cache TTL: $CACHE_TTL seconds"
echo "   - Max Memory: $MAX_MEMORY_MB MB"
echo "   - Canvas Rendering: $MAP_PREFER_CANVAS"
echo "   - Performance Metrics: $SHOW_PERFORMANCE_METRICS"
echo ""

# Start the Streamlit app
echo "🌐 Starting dashboard on http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --theme.base=light \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6" \
    --theme.textColor="#262730"