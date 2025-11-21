import os
import random
import warnings
import argparse
import yaml
import hashlib
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from meridian.data import load
from meridian.model import spec, model
from meridian.analysis import summarizer

# Get POC directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
POC_DIR = SCRIPT_DIR.parent.absolute()

SEED = 42
CONFIG_FILE = "configs/config_v1.yaml"


def setup_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    warnings.filterwarnings("ignore")


def load_config_and_data(config_file=CONFIG_FILE, data_file=None):
    # Resolve config file path relative to POC directory
    config_path = os.path.join(POC_DIR, config_file) if not os.path.isabs(config_file) else config_file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_name = os.getenv("DATASET_NAME", config.get("default_dataset", "weekly_applications"))
    model_config = config[model_name]

    # Override CSV path if data_file is provided
    if data_file:
        # Resolve data file path relative to POC directory
        if not os.path.isabs(data_file):
            csv_path = os.path.join(POC_DIR, "data/processed", data_file)
        else:
            csv_path = data_file
        # Ensure the file has .csv extension if not provided
        if not csv_path.endswith('.csv'):
            csv_path += '.csv'
        model_config["csv_path"] = csv_path
    else:
        # Use CSV path from config (resolve relative to POC directory)
        csv_path = model_config["csv_path"]
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(POC_DIR, csv_path)

    df = pd.read_csv(csv_path)
    time_col = model_config["columns"]["time"]
    df[time_col] = pd.to_datetime(df[time_col])

    display_cols = (
        [time_col, model_config["columns"].get("geo"), model_config["columns"]["kpi"]]
        + model_config["columns"]["media"][:3]
    )

    columns = model_config["columns"]
    coord_to_columns = load.CoordToColumns(
        time=columns["time"],
        geo=columns.get("geo"),
        kpi=columns["kpi"],
        media=columns["media"],
        media_spend=columns["media_spend"],
        controls=columns.get("controls", [])
    )

    media_to_channel = model_config["media_to_channel"]
    media_spend_to_channel = model_config["media_spend_to_channel"]
    feature_params = model_config.get("features", {})

    return {
        "df": df,
        "coord_to_columns": coord_to_columns,
        "model_config": model_config,
        "media_to_channel": media_to_channel,
        "media_spend_to_channel": media_spend_to_channel,
        "feature_params": feature_params,
        "display_cols": display_cols
    }


def build_model_and_sample(config_data):
    df = config_data["df"]
    coord_to_columns = config_data["coord_to_columns"]
    model_config = config_data["model_config"]
    media_to_channel = config_data["media_to_channel"]
    media_spend_to_channel = config_data["media_spend_to_channel"]
    feature_params = config_data["feature_params"]

    loader = load.DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type=model_config["kpi_type"],
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )
    meridian_data = loader.load()

    model_params = model_config["model"]
    model_spec = spec.ModelSpec(max_lag=model_params["max_lag"])

    for hyper in ["n_hidden_units", "n_fourier_nodes", "n_spline_knots"]:
        if hyper in model_params and hasattr(model_spec, hyper):
            setattr(model_spec, hyper, model_params[hyper])

    mmm = model.Meridian(input_data=meridian_data, model_spec=model_spec)

    if feature_params and hasattr(mmm, "set_feature_priors"):
        mmm.set_feature_priors(feature_params)

    sampling = model_config["sampling"]
    mmm.sample_posterior(
        n_chains=sampling["n_chains"],
        n_adapt=sampling["n_adapt"],
        n_burnin=sampling["n_burnin"],
        n_keep=sampling["n_keep"],
    )

    mmm.sample_prior(n_draws=1)
    return mmm, model_config


def compute_data_hash(df: pd.DataFrame, config_data: dict) -> str:
    """
    Computes a unique hash based on the data and model configuration.
    This allows uniquely identifying a model by its training data.
    """
    coord = config_data["coord_to_columns"]
    model_config = config_data["model_config"]

    # Data information
    data_info = {
        "shape": df.shape,
        "columns": sorted(df.columns.tolist()),
        "time_col": coord.time,
        "kpi_col": coord.kpi,
        "geo_col": coord.geo,
        "media_cols": sorted(coord.media) if coord.media else [],
        "media_spend_cols": sorted(coord.media_spend) if coord.media_spend else [],
        "date_range": {
            "start": str(df[coord.time].min()),
            "end": str(df[coord.time].max()),
        },
        # Value hash (sampled for performance)
        "data_hash": pd.util.hash_pandas_object(df).sum(),
    }

    # Model configuration
    model_info = {
        "kpi_type": model_config["kpi_type"],
        "model_params": model_config.get("model", {}),
        "sampling": model_config.get("sampling", {}),
    }

    # Create a combined string and compute hash
    combined_str = str(data_info) + str(model_info)
    data_hash = hashlib.sha256(combined_str.encode()).hexdigest()[:16]  # 16 characters is enough

    return data_hash


def generate_html_report(mmm, model_config, output_dir=None):
    """
    Generate the HTML report. If output_dir is specified,
    the report will be saved in that folder instead of the config path.
    """
    report = model_config["report"]

    if output_dir:
        output_html_path = os.path.join(output_dir, "report_data.html")
    else:
        output_html_path = os.path.join(POC_DIR, report["output_html"])

    mmm_summarizer = summarizer.Summarizer(mmm)
    mmm_summarizer.output_model_results_summary(
        output_html_path,
        filepath=".",
        start_date=report["start_date"],
        end_date=report["end_date"],
    )
    print(f"✓ HTML report generated: {output_html_path}")


def save_model_and_metadata(mmm, config_data, model_config, output_dir, data_hash, config_file=None, data_file=None):
    """
    Save the model and its metadata in the specified folder.
    """
    df = config_data["df"]
    coord = config_data["coord_to_columns"]

    # Extract the folder name (creation date)
    folder_name = os.path.basename(output_dir)

    # Save the model
    model_path = os.path.join(output_dir, "model.pkl")
    print(f"💾 Saving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(mmm, f)
    print("✓ Model saved successfully")

    # Save metadata
    now = datetime.now()
    metadata = {
        "folder_name": folder_name,
        "data_hash": data_hash,
        "created_at": now.isoformat(),
        "created_date": folder_name,  # Folder date, for easier retrieval
        "data_shape": df.shape,
        "data_columns": df.columns.tolist(),
        "date_range": {
            "start": str(df[coord.time].min()),
            "end": str(df[coord.time].max()),
        },
        "model_config": {
            "kpi_type": model_config["kpi_type"],
            "model_params": model_config.get("model", {}),
            "sampling": model_config.get("sampling", {}),
        },
        "config_file": config_file,
        "data_file": data_file,
    }

    metadata_path = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    print(f"✓ Metadata saved to: {metadata_path}")


def list_available_files(directory, extension=None):
    """List available files in a directory"""
    dir_path = os.path.join(POC_DIR, directory)
    if not os.path.exists(dir_path):
        return []

    files = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            if extension is None or file.endswith(extension):
                files.append(file)
    return sorted(files)


def interactive_select_config():
    """Interactive menu to select a configuration file"""
    configs = list_available_files("configs", extension=".yaml")

    if not configs:
        print("❌ No configuration file found in configs/")
        exit(1)

    print("\n" + "="*60)
    print("📋 CONFIGURATION FILE SELECTION")
    print("="*60)
    print("\nAvailable files:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}")

    while True:
        try:
            choice = input(f"\nChoose a file (1-{len(configs)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(configs):
                selected = configs[idx]
                print(f"✓ Selected configuration: {selected}\n")
                return selected
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(configs)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n❌ Canceled by user")
            exit(1)


def main_pipeline(config_file=None, data_file=None):
    setup_seed()
    config_data = load_config_and_data(config_file=config_file, data_file=data_file)

    # Create a folder name based on creation date
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n📅 Creation date: {date_folder}")

    # Compute data hash for metadata (but do not use it for name of folder)
    df = config_data["df"]
    data_hash = compute_data_hash(df, config_data)

    # Create output folder organized by creation date
    output_dir = os.path.join(POC_DIR, "outputs", "models", date_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output folder: {output_dir}\n")

    # Build and train model
    mmm, model_config = build_model_and_sample(config_data)

    # Save model and metadata
    save_model_and_metadata(
        mmm, config_data, model_config,
        output_dir, data_hash,
        config_file=config_file,
        data_file=data_file
    )

    # Generate HTML report in the same folder
    print("\n📄 Generating HTML report...")
    generate_html_report(mmm, model_config, output_dir=output_dir)

    print(f"\n✅ Model and report saved in: {output_dir}")
    print(f"   🤖 Model: model.pkl")
    print(f"   📄 Report: report_data.html")
    print(f"   📋 Metadata: metadata.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Meridian MMM model with configuration selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (config selection)
  python run.py

  # Specify a config file directly
  python run.py --config config_v1.yaml

  # List available configuration files
  python run.py --list-configs
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file name in configs/ (e.g. config_v1.yaml). Default: interactive mode"
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Display the list of available configuration files"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # List available files if requested
    if args.list_configs:
        configs = list_available_files("configs", extension=".yaml")
        print("\nAvailable configuration files in configs/:")
        for config in configs:
            print(f"  - {config}")
        print()
        exit(0)

    # Determine config file
    if args.config:
        # Command line mode: use provided config
        config_file = f"configs/{args.config}"
        if not config_file.endswith('.yaml') and not config_file.endswith('.yml'):
            config_file += '.yaml'

        # Check if config file exists
        config_path = os.path.join(POC_DIR, config_file) if not os.path.isabs(config_file) else config_file
        if not os.path.exists(config_path):
            print(f"❌ Error: The configuration file '{config_file}' does not exist.")
            print(f"\nAvailable files:")
            configs = list_available_files("configs", extension=".yaml")
            for config in configs:
                print(f"  - {config}")
            exit(1)

        print(f"📋 Configuration: {config_file}")
        print(f"📊 Data: (uses the one from the config file)\n")
    else:
        # Interactive mode: select config file
        print("\n" + "="*60)
        print("🚀 LAUNCHING MERIDIAN MMM MODEL")
        print("="*60)

        selected_config = interactive_select_config()
        config_file = f"configs/{selected_config}"

        print("="*60)
        print("🔄 LAUNCHING MODEL...")
        print("="*60 + "\n")

    # Run pipeline (always use dataset from config, so data_file=None)
    main_pipeline(config_file=config_file, data_file=None)

