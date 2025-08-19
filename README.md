# 🇪🇹 Ethiopia Violence Analysis Dashboard

Interactive web dashboard for analyzing violence patterns across Ethiopian administrative units using ACLED data and population rasters. The dashboard provides comprehensive analysis at multiple administrative levels (Regions, Zones, and Woredas) with interactive visualizations and statistical insights.

## Features

- 🗺️ **Interactive Map Visualization** with Folium
  - Administrative units analysis (Regions/Zones)
  - Individual woreda classification
  - Color-coded violence intensity levels
  - Interactive popups with detailed statistics

- 📊 **Real-time Parameter Adjustment**
  - Death rate thresholds (per 100k population)
  - Minimum absolute death thresholds
  - Aggregation thresholds for administrative units
  - Flexible 12-month analysis periods

- 📈 **Comprehensive Statistical Analysis**
  - Violence-affected areas ranking
  - Population vs. violence deaths correlation
  - Distribution of violence levels
  - Woreda classification breakdown

- 📱 **Responsive Design** with modern UI
- 💾 **Data Export Functionality** (CSV format)
- 🎯 **Multi-level Administrative Analysis** (ADM1, ADM2, ADM3)

## Data Sources

- **Ethiopian Administrative Boundaries**: Single comprehensive shapefile with all administrative levels
- **Population Raster Data**: WorldPop 2020 population estimates
- **ACLED Conflict Data**: Armed Conflict Location & Event Data Project

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone this repository**
   ```bash
   git clone <repository-url>
   cd ethiopia-dashboard
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv eth_dashboard_env
   # On Windows:
   eth_dashboard_env\Scripts\Activate.ps1
   # On macOS/Linux:
   source eth_dashboard_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your data files to the `data/` directory** (see Data Structure below)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard** at `http://localhost:8501`

## Data Structure

```
data/
├── ETH/
│   ├── Ethiopia.shp          # Main shapefile with all admin levels
│   ├── Ethiopia.dbf          # Attribute data
│   ├── Ethiopia.shx          # Shape index
│   ├── Ethiopia.prj          # Projection file
│   └── Ethiopia.shp.xml      # Metadata
├── processed/
│   └── intersection_result_acled.csv  # ACLED conflict data
└── eth_ppp_2020.tif          # Population raster (optional)
```

### Administrative Levels
- **ADM1 (Regions)**: 13 regions (e.g., Tigray, Amhara, Oromia)
- **ADM2 (Zones)**: 92 zones (administrative subdivisions)
- **ADM3 (Woredas)**: 1082 woredas (lowest administrative level)

## Usage Guide

### 1. **Violence Classification Parameters**
   - **Death Rate Threshold**: Minimum deaths per 100,000 population
   - **Min Deaths Threshold**: Absolute minimum number of deaths
   - **Aggregation Threshold**: Share of woredas affected to mark unit as high-violence

### 2. **Analysis Settings**
   - **Analysis Period**: Select 12-month periods (calendar year or mid-year cycles)
   - **Administrative Level**: Choose between Regions (ADM1) or Zones (ADM2)
   - **Map Variable**: Display share of woredas affected or share of population affected

### 3. **Interactive Features**
   - **Administrative Units Map**: Aggregated analysis by regions/zones
   - **Woreda Classification Map**: Individual woreda violence classification
   - **Supporting Charts**: Statistical analysis and insights
   - **Data Export**: Download results in CSV format

### 4. **Key Metrics Displayed**
   - High violence units count and percentage
   - Affected woredas and population statistics
   - Total deaths and national death rates
   - Violence coverage and population impact

## Deployment

### Local Development
The app is configured for local development with Streamlit. Simply run:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
This app is configured for Streamlit Cloud deployment:
1. Push your code to GitHub
2. Connect your repository via [share.streamlit.io](https://share.streamlit.io)
3. Deploy with the following configuration:
   - **Main file path**: `app.py`
   - **Python version**: 3.8+

### Requirements for Deployment
- All data files must be included in the repository
- Virtual environment files are excluded via `.gitignore`
- Dependencies are specified in `requirements.txt`

## Technical Details

### Key Libraries
- **Streamlit**: Web application framework
- **GeoPandas**: Geospatial data manipulation
- **Folium**: Interactive mapping
- **Plotly**: Statistical visualizations
- **Rasterio**: Raster data processing
- **Pandas**: Data manipulation and analysis

### Data Processing
- **Population Estimation**: Raster-based population extraction for each woreda
- **Boundary Aggregation**: Administrative levels created by dissolving woreda boundaries
- **Conflict Data Integration**: ACLED data matched to administrative units
- **Statistical Analysis**: Real-time calculation of violence metrics and thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on GitHub or contact the development team.