# ROIxplain - Media Mix Modeling (MMM) Framework

A comprehensive Media Mix Modeling solution built on **Google Meridian** for analyzing marketing channel effectiveness, calculating ROI, and generating actionable insights for budget optimization.

## 📋 Overview

ROIxplain is an end-to-end Media Mix Modeling (MMM) framework that leverages Google's Meridian library to:

- **Measure channel impact**: Quantify the contribution of each marketing channel (TikTok, Facebook, Google Ads, etc.)
- **Calculate ROI**: Compute Return on Investment metrics per channel
- **Generate insights**: Produce visual reports with actionable recommendations
- **Optimize budgets**: Provide data-driven guidance for marketing budget allocation

### What is Media Mix Modeling?

Media Mix Modeling (MMM) is a statistical analysis technique that uses historical data to quantify the impact of various marketing channels on business outcomes. Unlike attribution models, MMM uses aggregate data and can account for external factors, seasonality, and carryover effects.

### About Google Meridian

This project is built on **[Google Meridian](https://github.com/google/meridian)**, an open-source Bayesian Media Mix Modeling framework developed by Google. Meridian provides:

- **Bayesian hierarchical models** for robust parameter estimation
- **MCMC sampling** (Markov Chain Monte Carlo) for uncertainty quantification
- **Flexible feature engineering** with splines, Fourier terms, and lag effects
- **Comprehensive reporting** with built-in visualization capabilities

📚 **Official Documentation**: [Google Meridian GitHub](https://github.com/google/meridian)  
📖 **Research Paper**: [Meridian: A Bayesian Framework for Media Mix Modeling](https://research.google/pubs/pub45901/)

## 📊 Dataset

This project uses the **Marketing Mix Dataset** from Kaggle:

🔗 **Source**: [Marketing Mix Dataset - Kaggle](https://www.kaggle.com/datasets/orosas/marketing-mix-dataset?resource=download)

The dataset contains weekly aggregated data including:
- Marketing spend by channel (TikTok, Facebook, Google Ads)
- Sales/revenue metrics
- Temporal and geographic dimensions
- Control variables (optional)

## 🏗️ Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                        │
│  ┌──────────────┐                                               │
│  │  Raw CSV     │  →  Processed CSV (data/processed/)           │
│  │  (Kaggle)    │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION LOADING                      │
│  ┌──────────────┐                                               │
│  │  YAML Config │  →  Column mapping, model params, sampling    │
│  │  (configs/)  │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE MERIDIAN PIPELINE                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. DataLoader: DataFrameDataLoader                      │   │
│  │     - Load CSV data                                      │   │
│  │     - Map columns (time, geo, KPI, media, spend)         │   │
│  │     - Create Meridian data structure                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  2. Model Specification: ModelSpec                       │   │
│  │     - max_lag: Carryover effect window                   │   │
│  │     - n_hidden_units: Neural network capacity            │   │
│  │     - n_fourier_nodes: Seasonality modeling              │   │
│  │     - n_spline_knots: Trend flexibility                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  3. Model Training: Meridian                             │   │
│  │     - Initialize Bayesian model                          │   │
│  │     - Set feature priors (Beta distributions)            │   │
│  │     - MCMC sampling (n_chains, n_adapt, n_burnin, n_keep)│   │
│  │     - Posterior inference                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT GENERATION                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ model.pkl    │  │ metadata.yaml│  │ report_data  │           │
│  │ (Trained     │  │ (Config,     │  │ .html        │           │
│  │  Model)      │  │  Hash, Date) │  │ (Meridian    │           │
│  └──────────────┘  └──────────────┘  │  Report)     │           │
│                                      └──────────────┘           │
│                            ↓                                    │
│                    ┌──────────────┐                             │
│                    │ custom_report│                             │
│                    │ .html        │                             │
│                    │ (Enhanced    │                             │ 
│                    │  Insights)   │                             │
│                    └──────────────┘                             │ 
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Virtual environment**: Recommended (included in setup)

### Step-by-Step Installation

1. **Navigate to the project directory**
   ```bash
   cd mmm_meridian
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

4. **Upgrade pip and install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify installation**
   ```bash
   python scripts/setup_check.py
   ```

   This script verifies:
   - Python version compatibility
   - All required packages installation
   - Directory structure
   - Configuration files validity
   - Data files accessibility
   - Google Meridian module import

## 📁 Project Structure

```
mmm_meridian/
├── configs/                    # YAML configuration files
│   ├── config_v1.yaml          # Primary configuration
│   └── config_v2.yaml          # Alternative configuration
│
├── data/                       # Data directory
│   ├── raw/                    # Raw data files (from Kaggle)
│   │   └── data.csv
│   └── processed/              # Processed/prepared data
│       └── data_processed.csv
│
├── scripts/                    # Python scripts
│   ├── run.py                  # Main training script
│   ├── save_model.py           # Model persistence with metadata
│   ├── custom_report.py        # Enhanced report generation
│   └── setup_check.py          # Environment verification
│
├── outputs/                    # Generated outputs
│   └── models/                 # Trained models (organized by timestamp)
│       └── YYYY-MM-DD_HH-MM-SS/
│           ├── model.pkl       # Serialized Meridian model
│           ├── metadata.yaml   # Model metadata and configuration
│           ├── report_data.html # Meridian technical report
│           └── custom_report.html # Enhanced marketing report
│
├── notebook/                   # Jupyter notebooks
│   └── data_exploration.ipynb  # Data exploration and analysis
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🎯 Usage

> **⚠️ Important**: All scripts must be run from the **mmm_meridian directory** with the virtual environment activated.

### Quick Start

1. **Navigate to mmm_meridian directory**:
   ```bash
   cd mmm_meridian
   ```

2. **Activate the virtual environment**:
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

3. **Run scripts from mmm_meridian directory**:
   ```bash
   # Run directly with Python
   python scripts/run.py
   ```

### 1. Train a Model

**Interactive mode** (selects configuration file interactively):
```bash
cd mmm_meridian
source .venv/bin/activate  # macOS/Linux
python scripts/run.py
```

**Specify configuration file**:
```bash
cd mmm_meridian
source .venv/bin/activate
python scripts/run.py --config config_v1.yaml
```

**List available configurations**:
```bash
cd mmm_meridian
source .venv/bin/activate
python scripts/run.py --list-configs
```

**What happens during training**:
1. Loads configuration from YAML file
2. Reads and validates CSV data
3. Initializes Google Meridian model with specified parameters
4. Performs MCMC sampling (may take several minutes to hours depending on data size)
5. Generates technical report using Meridian's built-in summarizer
6. Saves model, metadata, and reports to timestamped directory

### 2. Save Model with Metadata

**Basic usage**:
```bash
python scripts/save_model.py --config config_v1.yaml
```

**Specify custom data file**:
```bash
python scripts/save_model.py --config config_v1.yaml --data data_processed.csv
```

**List saved models**:
```bash
python scripts/save_model.py --list
```

**Features**:
- Automatic timestamp-based directory creation
- Data hash computation for model identification
- Complete metadata capture (configuration, data shape, date ranges)
- Model persistence for future analysis

### 3. Generate Custom Marketing Report

```bash
python scripts/custom_report.py
```

**Interactive workflow**:
1. Select a saved model from the list
2. Automatically extracts:
   - R² score and model quality assessment
   - ROI metrics per channel
   - Model fit visualizations
   - Channel contribution charts
3. Generates actionable marketing insights:
   - Channel performance recommendations
   - Budget reallocation suggestions
   - Optimization opportunities
   - Risk assessments

**Report contents**:
- **Model Performance**: R² score with quality interpretation
- **ROI by Channel**: Return on investment for each marketing channel
- **Model Fit Visualization**: Predicted vs. actual revenue comparison
- **Channel Contribution**: Breakdown of each channel's impact
- **Actionable Insights**: Data-driven recommendations for marketing teams

## ⚙️ Configuration

Configuration files (`configs/*.yaml`) define all aspects of the modeling process:

### Configuration Structure

```yaml
default_dataset: weekly_applications

weekly_applications:
  # Data source
  csv_path: "data/processed/data_processed.csv"
  kpi_type: "revenue"  # or "conversions"
  
  # Column mapping
  columns:
    time: "Date"                    # Temporal dimension
    geo: "Geo"                      # Geographic dimension (optional)
    kpi: "Sales"                    # Key Performance Indicator
    media:                          # Media channels (impressions/activity)
      - "TikTok"
      - "Facebook"
      - "Google Ads"
    media_spend:                    # Media spend (budget)
      - "TikTok"
      - "Facebook"
      - "Google Ads"
    controls: []                    # Control variables (optional)
  
  # Channel mapping
  media_to_channel:
    "TikTok": "TikTok"
    "Facebook": "Facebook"
    "Google Ads": "Google Ads"
  
  # Feature priors (Beta distribution parameters)
  features:
    "TikTok":
      distribution: "beta"
      alpha: 1.10
      beta: 1.15
  
  # Model hyperparameters
  model:
    max_lag: 12              # Carryover effect window (weeks)
    n_hidden_units: 32       # Neural network capacity
    n_fourier_nodes: 12      # Seasonality modeling complexity
    n_spline_knots: 16       # Trend flexibility
  
  # MCMC sampling parameters
  sampling:
    n_chains: 6              # Number of parallel chains
    n_adapt: 4000            # Adaptation iterations
    n_burnin: 800            # Burn-in iterations
    n_keep: 2000             # Posterior samples to keep
  
  # Report configuration
  report:
    start_date: "2018-01-07"
    end_date: "2021-10-31"
    output_html: "outputs/report_data.html"
```

### Key Parameters Explained

**Model Parameters** (Google Meridian):
- `max_lag`: Maximum number of periods for carryover effects. Higher values capture longer-term impacts but increase complexity.
- `n_hidden_units`: Number of hidden units in the neural network component. Controls model flexibility.
- `n_fourier_nodes`: Number of Fourier basis functions for seasonality. Higher values capture more complex seasonal patterns.
- `n_spline_knots`: Number of knots for spline-based trend modeling. Controls trend flexibility.

**MCMC Parameters**:
- `n_chains`: Number of independent MCMC chains. More chains improve convergence diagnostics.
- `n_adapt`: Adaptation phase length. Allows the sampler to tune step sizes.
- `n_burnin`: Burn-in iterations discarded before sampling. Removes initialization bias.
- `n_keep`: Number of posterior samples retained. More samples improve uncertainty estimates.

📚 **For detailed parameter guidance, refer to**: [Google Meridian Documentation](https://github.com/google/meridian)

## 📊 Outputs & Results

Each trained model generates a complete analysis package:

### Generated Files

1. **`model.pkl`**
   - Serialized Meridian model object
   - Contains all posterior samples and model state
   - Can be loaded for future predictions or analysis

2. **`metadata.yaml`**
   - Model creation timestamp
   - Data hash (unique identifier for training data)
   - Data shape and column information
   - Date range of training data
   - Complete model configuration snapshot
   - Configuration file reference

3. **`report_data.html`**
   - Technical report generated by Google Meridian
   - Includes model diagnostics, parameter estimates, and visualizations
   - Uses Meridian's built-in `Summarizer` class

4. **`custom_report.html`**
   - Enhanced marketing-focused report
   - Extracts and visualizes key business metrics
   - Provides actionable recommendations
   - Modern, interactive HTML interface

### Key Metrics

**R² Score (Coefficient of Determination)**
- Measures model fit quality (0 to 1)
- Interpretation:
  - **≥ 0.75**: Excellent fit
  - **0.50 - 0.75**: Good fit, room for improvement
  - **< 0.50**: Needs improvement

**ROI (Return on Investment)**
- Calculated per channel: Revenue generated per $1 spent
- Values > 1.0 indicate profitable channels
- Used for budget allocation decisions

**Channel Contribution**
- Relative impact of each channel on total sales
- Includes baseline (non-marketing) contribution
- Visualized in contribution charts

**Actionable Insights**
- Channel performance rankings
- Budget reallocation recommendations
- Optimization opportunities
- Risk assessments

## 🔧 Dependencies

### Core Dependencies

- **numpy** (≥1.21.0): Numerical computing
- **pandas** (≥1.3.0): Data manipulation and analysis
- **tensorflow** (≥2.8.0): Deep learning framework
- **tensorflow-probability** (≥0.16.0): Probabilistic modeling
- **google-meridian** (latest): Google's MMM framework
- **PyYAML** (≥5.4.0): Configuration file parsing
- **scipy** (≥1.7.0): Scientific computing
- **matplotlib** (≥3.4.0): Visualization
- **Jinja2** (≥3.0.0): Template engine
- **joblib** (≥1.0.0): Parallel processing

See `requirements.txt` for complete dependency list with versions.

### Google Meridian

The project relies on **[Google Meridian](https://github.com/google/meridian)** for the core MMM functionality:

- **Installation**: `pip install google-meridian`
- **Documentation**: [Meridian GitHub Repository](https://github.com/google/meridian)
- **Key Modules Used**:
  - `meridian.data.load`: Data loading and preprocessing
  - `meridian.model.spec`: Model specification
  - `meridian.model.model`: Model training and inference
  - `meridian.analysis.summarizer`: Report generation

## 🐛 Troubleshooting

### Environment Verification

Run the setup check script to diagnose issues:
```bash
python scripts/setup_check.py
```

This comprehensive check verifies:
- ✅ Python version compatibility
- ✅ Package installations
- ✅ Directory structure
- ✅ Configuration file validity
- ✅ Data file accessibility
- ✅ Google Meridian import

### Common Issues

**Issue: Google Meridian import error**
```bash
# Solution: Reinstall Meridian
pip install --upgrade google-meridian
```

**Issue: Virtual environment path incorrect**
- Ensure you're using `.venv` in the `mmm_meridian/` directory
- Correct path: `/path/to/mmm_meridian/.venv`
- Activate with: `source mmm_meridian/.venv/bin/activate`

**Issue: Data file not found**
- Verify CSV path in `configs/*.yaml` is relative to `mmm_meridian/` directory
- Check that `data/processed/data_processed.csv` exists
- Use absolute paths if needed

**Issue: MCMC sampling takes too long**
- Reduce `n_keep` in configuration (fewer samples)
- Reduce `n_chains` (fewer parallel chains)
- Reduce `n_adapt` (shorter adaptation phase)
- Note: This may reduce model quality

**Issue: Model convergence warnings**
- Increase `n_adapt` (longer adaptation)
- Increase `n_burnin` (more burn-in iterations)
- Check data quality and preprocessing
- Review model hyperparameters

**Issue: Low R² score**
- Review data quality and completeness
- Add control variables if available
- Adjust model hyperparameters (increase `n_hidden_units`, `n_fourier_nodes`)
- Check for data preprocessing issues
- Consider longer time series

## 📚 Additional Resources

### Google Meridian

- **GitHub Repository**: [google/meridian](https://github.com/google/meridian)
- **Documentation**: Check the repository's README and examples
- **Research Paper**: [Meridian: A Bayesian Framework for Media Mix Modeling](https://research.google/pubs/pub45901/)

### Media Mix Modeling

- **Wikipedia**: [Media Mix Modeling](https://en.wikipedia.org/wiki/Media_mix_modeling)
- **Kaggle Dataset**: [Marketing Mix Dataset](https://www.kaggle.com/datasets/orosas/marketing-mix-dataset?resource=download)

### Bayesian Statistics & MCMC

- **TensorFlow Probability**: [Documentation](https://www.tensorflow.org/probability)
- **MCMC Methods**: Understanding Markov Chain Monte Carlo sampling

## 📝 Notes

- **Model Persistence**: Models are saved with timestamps for easy tracking and versioning
- **Data Hashing**: Each model includes a hash of training data for identification
- **Reproducibility**: Seed is set to 42 for reproducible results
- **Report Viewing**: HTML reports can be opened directly in any web browser
- **Scalability**: The framework can handle multiple datasets and configurations

## 📄 License

This project uses publicly available data from Kaggle. Please review the dataset's license terms on the Kaggle page.

The Google Meridian library is subject to its own license terms. Please refer to the [Meridian repository](https://github.com/google/meridian) for details.

---

