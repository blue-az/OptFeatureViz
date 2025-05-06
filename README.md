# Optiver Trading at the Close - Data Visualization Dashboard

This project provides an interactive Streamlit dashboard to explore and visualize the training data (`train.csv`) from the Kaggle competition "Optiver - Trading at the Close".

## Purpose

The dashboard allows users to quickly filter the large training dataset by stock ID and date range, and then visualize key aspects of the data, such as target distributions, intraday price movements, and relationships between different order book variables. This aids in exploratory data analysis (EDA) and understanding the dataset's characteristics.

## Features

*   **Data Loading:** Loads the `train.csv` file.
*   **Memory Optimization:** Applies basic data type downcasting to reduce memory usage.
*   **Interactive Filtering:**
    *   Select one or multiple Stock IDs.
    *   Select a Date ID range using a slider.
*   **Data Overview:** Displays the shape and first few rows of the filtered data.
*   **Visualizations (using Plotly Express):**
    *   **Target Distribution:** Histogram showing the distribution of the `target` variable.
    *   **Intraday Time Series:** Line plot showing a selected variable (e.g., `wap`, `target`, `liquidity_imbalance`) over `seconds_in_bucket`, colored by Stock ID. Subsamples data for performance if the filtered dataset is large.
    *   **Relationship Explorer:** Scatter plot allowing selection of X and Y variables to explore correlations visually. Subsamples data for performance.
    *   **Aggregate View:** Box plot showing the distribution of a selected variable grouped by Stock ID.

## Prerequisites

*   **Python:** Version 3.8 or higher recommended.
*   **pip:** Python package installer.
*   **Data:** The `train.csv` file from the [Optiver - Trading at the Close Kaggle competition](https://www.kaggle.com/competitions/optiver-trading-at-the-close).

## Setup

1.  **Clone or Download:** Get the `dashboard.py` script.
2.  **Place Data:** Place the `train.csv` file in the same directory as `dashboard.py`, or create a subdirectory (e.g., `data/`) and place it there. If you use a subdirectory, update the `DATA_FILE` variable at the top of `dashboard.py` accordingly (e.g., `DATA_FILE = "./data/train.csv"`).
3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
4.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas numpy plotly
    ```
    *(Note: `lightgbm` is not strictly required for this dashboard script but might be in your environment already).*

## How to Run

1.  **Navigate to Directory:** Open your terminal or command prompt and navigate to the directory containing `dashboard.py` and the `train.csv` file (or its parent directory if `train.csv` is in a subdirectory).
2.  **Activate Environment:** If you created a virtual environment, activate it:
    ```bash
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Run Streamlit App:**
    ```bash
    streamlit run dashboard.py
    ```
4.  **View in Browser:** Streamlit will start a local server and usually open the dashboard automatically in your web browser. If not, navigate to the URL provided in the terminal (typically `http://localhost:8501`).
5.  **Interact:** Use the sidebar controls to filter the data by stock and date range. Explore the different visualization tabs.

## Notes

*   The `train.csv` file is very large (several gigabytes). Loading and filtering may take some time, especially on the first run before data is cached by Streamlit (`@st.cache_data`).
*   Visualizations involving large amounts of filtered data (Time Series, Scatter Plot) use subsampling to maintain browser responsiveness.
*   The memory reduction applied is basic; more aggressive techniques could be used if needed.

## Potential Future Enhancements

*   Add more sophisticated visualizations (e.g., correlation heatmaps, pair plots).
*   Incorporate generated features from the modeling scripts for analysis.
*   Add options for different time aggregations (e.g., daily summaries).
*   Implement more advanced caching or data loading strategies for very large datasets.
