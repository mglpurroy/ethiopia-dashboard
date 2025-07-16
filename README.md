Description# 🇪🇹 Ethiopia Violence Analysis Dashboard

Interactive web dashboard for analyzing violence patterns across Ethiopian administrative units using ACLED data and population rasters.

## Features

- 🗺️ Interactive map visualization with Folium
- 📊 Real-time parameter adjustment
- 📈 Statistical analysis and charts
- 📱 Responsive design
- 💾 Data export functionality

## Data Sources

- Ethiopian Administrative Boundaries (CSA)
- Population Raster Data (WorldPop)
- ACLED Conflict Data

## Setup Instructions

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your data files to the `data/` directory
4. Run: `streamlit run app.py`

## Deployment

This app is configured for Streamlit Cloud deployment. Simply push to GitHub and connect via [share.streamlit.io](https://share.streamlit.io).

## Data Structure

```
data/
├── processed/
│   └── intersection_result_acled.csv
├── eth_ppp_2020.tif
└── eth_adm_csa_bofedb_2021_shp/
    ├── eth_admbnda_adm1_csa_bofedb_2021.shp
    ├── eth_admbnda_adm2_csa_bofedb_2021.shp
    └── eth_admbnda_adm3_csa_bofedb_2021.shp
```

## Usage

1. Adjust violence classification parameters in the sidebar
2. Select analysis year and administrative level
3. Explore the interactive map
4. Review metrics and supporting charts
5. Export results as needed