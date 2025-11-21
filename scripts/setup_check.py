#!/usr/bin/env python3
"""
ROIxplain Environment Check Script.
Verifies that all dependencies, files, and folders are properly configured.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import List, Tuple, Dict


class Colors:
    """Terminal output colors"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class CheckResult:
    """Result of a check"""
    def __init__(self, name: str, status: bool, message: str = "", details: str = ""):
        self.name = name
        self.status = status
        self.message = message
        self.details = details


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_result(result: CheckResult):
    """Print a single check result"""
    status_icon = f"{Colors.GREEN}✓{Colors.END}" if result.status else f"{Colors.RED}✗{Colors.END}"
    status_text = f"{Colors.GREEN}OK{Colors.END}" if result.status else f"{Colors.RED}ERROR{Colors.END}"
    
    print(f"  {status_icon} {result.name}: {status_text}")
    if result.message:
        print(f"    {result.message}")
    if result.details:
        print(f"    {Colors.YELLOW}ℹ{Colors.END} {result.details}")


def check_python_version() -> CheckResult:
    """Check Python version"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        return CheckResult(
            "Python Version",
            True,
            f"Python {version_str}",
            "Compatible version (>= 3.8 required)"
        )
    else:
        return CheckResult(
            "Python Version",
            False,
            f"Python {version_str}",
            "Python 3.8 or higher required"
        )


def check_package(package_name: str, import_name: str = None, version_attr: str = "__version__") -> CheckResult:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = "?"
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
        elif hasattr(module, "version"):
            version = getattr(module, "version")
        
        return CheckResult(
            package_name,
            True,
            f"Version {version}",
            ""
        )
    except ImportError as e:
        return CheckResult(
            package_name,
            False,
            "Not installed",
            f"Install with: pip install {package_name}"
        )


def check_required_packages() -> List[CheckResult]:
    """Check all required packages"""
    print_header("PYTHON DEPENDENCIES CHECK")
    
    # Essential packages for Meridian MMM
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tensorflow", "tensorflow"),
        ("tensorflow-probability", "tensorflow_probability"),
        ("meridian", "meridian"),
        ("PyYAML", "yaml"),
        ("Jinja2", "jinja2"),
        ("joblib", "joblib"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
    ]
    
    results = []
    for package_name, import_name in packages:
        result = check_package(package_name, import_name)
        results.append(result)
        print_result(result)
    
    return results


def check_directory_structure() -> List[CheckResult]:
    """Check required folder structure"""
    print_header("DIRECTORY STRUCTURE CHECK")
    
    script_dir = Path(__file__).parent
    poc_dir = script_dir.parent
    
    required_dirs = {
        "configs": poc_dir / "configs",
        "data/raw": poc_dir / "data" / "raw",
        "data/processed": poc_dir / "data" / "processed",
        "outputs": poc_dir / "outputs",
        "outputs/models": poc_dir / "outputs" / "models",
        "scripts": poc_dir / "scripts",
        "notebook": poc_dir / "notebook",
    }
    
    results = []
    for name, path in required_dirs.items():
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        
        if exists and is_dir:
            # Check write permissions for outputs directories
            if "outputs" in name:
                writable = os.access(path, os.W_OK)
                if writable:
                    results.append(CheckResult(
                        f"Directory {name}",
                        True,
                        str(path),
                        "Directory exists and is writable"
                    ))
                else:
                    results.append(CheckResult(
                        f"Directory {name}",
                        False,
                        str(path),
                        "Directory exists but is not writable"
                    ))
            else:
                results.append(CheckResult(
                    f"Directory {name}",
                    True,
                    str(path),
                    ""
                ))
        else:
            results.append(CheckResult(
                f"Directory {name}",
                False,
                str(path),
                "Directory missing - please create this directory"
            ))
        
        print_result(results[-1])
    
    return results


def check_config_files() -> List[CheckResult]:
    """Check configuration files"""
    print_header("CONFIGURATION FILES CHECK")
    
    script_dir = Path(__file__).parent
    configs_dir = script_dir.parent / "configs"
    
    results = []
    
    # Check if configs directory exists
    if not configs_dir.exists():
        results.append(CheckResult(
            "Configs directory",
            False,
            "Configs directory missing",
            ""
        ))
        print_result(results[-1])
        return results
    
    # List configuration files
    config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    
    if not config_files:
        results.append(CheckResult(
            "Configuration files",
            False,
            "No YAML file found in configs/",
            "Create at least one config_v1.yaml file"
        ))
        print_result(results[-1])
        return results
    
    # Check each configuration file
    for config_file in sorted(config_files):
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check basic structure
            has_default = "default_dataset" in config if config else False
            has_datasets = len(config) > 1 if config else False
            
            if has_default and has_datasets:
                results.append(CheckResult(
                    f"Config: {config_file.name}",
                    True,
                    f"Valid file ({len(config)-1} dataset(s))",
                    f"Default dataset: {config.get('default_dataset', 'N/A')}"
                ))
            else:
                results.append(CheckResult(
                    f"Config: {config_file.name}",
                    False,
                    "Invalid structure",
                    "Must include 'default_dataset' and at least one dataset"
                ))
        except Exception as e:
            results.append(CheckResult(
                f"Config: {config_file.name}",
                False,
                f"Error while reading: {str(e)}",
                ""
            ))
        
        print_result(results[-1])
    
    return results


def check_data_files() -> List[CheckResult]:
    """Check data files"""
    print_header("DATA FILES CHECK")
    
    script_dir = Path(__file__).parent
    poc_dir = script_dir.parent
    
    data_raw_dir = poc_dir / "data" / "raw"
    data_processed_dir = poc_dir / "data" / "processed"
    
    results = []
    
    # Check raw files
    if data_raw_dir.exists():
        raw_files = list(data_raw_dir.glob("*.csv"))
        if raw_files:
            for data_file in raw_files:
                try:
                    import pandas as pd
                    df = pd.read_csv(data_file)
                    results.append(CheckResult(
                        f"Raw data: {data_file.name}",
                        True,
                        f"{len(df)} rows, {len(df.columns)} columns",
                        f"Size: {data_file.stat().st_size / 1024:.1f} KB"
                    ))
                except Exception as e:
                    results.append(CheckResult(
                        f"Raw data: {data_file.name}",
                        False,
                        f"Error while reading: {str(e)}",
                        ""
                    ))
                print_result(results[-1])
        else:
            results.append(CheckResult(
                "Raw data",
                False,
                "No CSV file found in data/raw/",
                ""
            ))
            print_result(results[-1])
    else:
        results.append(CheckResult(
            "data/raw directory",
            False,
            "Directory missing",
            ""
        ))
        print_result(results[-1])
    
    # Check processed files
    if data_processed_dir.exists():
        processed_files = list(data_processed_dir.glob("*.csv"))
        if processed_files:
            for data_file in processed_files:
                try:
                    import pandas as pd
                    df = pd.read_csv(data_file)
                    results.append(CheckResult(
                        f"Processed data: {data_file.name}",
                        True,
                        f"{len(df)} rows, {len(df.columns)} columns",
                        f"Size: {data_file.stat().st_size / 1024:.1f} KB"
                    ))
                except Exception as e:
                    results.append(CheckResult(
                        f"Processed data: {data_file.name}",
                        False,
                        f"Error while reading: {str(e)}",
                        ""
                    ))
                print_result(results[-1])
        else:
            results.append(CheckResult(
                "Processed data",
                False,
                "No CSV file found in data/processed/",
                ""
            ))
            print_result(results[-1])
    else:
        results.append(CheckResult(
            "data/processed directory",
            False,
            "Directory missing",
            ""
        ))
        print_result(results[-1])
    
    return results


def check_scripts() -> List[CheckResult]:
    """Check required Python scripts"""
    print_header("SCRIPTS CHECK")
    
    script_dir = Path(__file__).parent
    
    required_scripts = [
        "run.py",
        "save_model.py",
        "custom_report.py",
    ]
    
    results = []
    for script_name in required_scripts:
        script_path = script_dir / script_name
        if script_path.exists():
            # Check that it is a valid Python file
            try:
                with open(script_path, 'r') as f:
                    content = f.read()
                    # Basic check
                    if "import" in content or "from" in content:
                        results.append(CheckResult(
                            f"Script: {script_name}",
                            True,
                            "Valid Python file",
                            ""
                        ))
                    else:
                        results.append(CheckResult(
                            f"Script: {script_name}",
                            False,
                            "File seems empty or invalid",
                            ""
                        ))
            except Exception as e:
                results.append(CheckResult(
                    f"Script: {script_name}",
                    False,
                    f"Error while reading: {str(e)}",
                    ""
                ))
        else:
            results.append(CheckResult(
                f"Script: {script_name}",
                False,
                "File missing",
                ""
            ))
        
        print_result(results[-1])
    
    return results


def check_meridian_import() -> CheckResult:
    """Check import of Meridian and its modules"""
    print_header("MERIDIAN MODULE CHECK")
    
    try:
        from meridian.data import load
        from meridian.model import spec, model
        from meridian.analysis import summarizer, visualizer
        
        return CheckResult(
            "Meridian Module",
            True,
            "Meridian imported successfully",
            "Available modules: data.load, model, analysis"
        )
    except ImportError as e:
        return CheckResult(
            "Meridian Module",
            False,
            f"Import error: {str(e)}",
            "Install with: pip install google-meridian"
        )


def generate_summary(all_results: Dict[str, List[CheckResult]]):
    """Generate a summary of all checks"""
    print_header("CHECK SUMMARY")
    
    total_checks = 0
    passed_checks = 0
    
    for category, results in all_results.items():
        for result in results:
            total_checks += 1
            if result.status:
                passed_checks += 1
    
    success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    print(f"\n  {Colors.BOLD}Results:{Colors.END}")
    print(f"    ✓ Passed: {Colors.GREEN}{passed_checks}{Colors.END}")
    print(f"    ✗ Failed: {Colors.RED}{total_checks - passed_checks}{Colors.END}")
    print(f"    📊 Total: {total_checks}")
    print(f"    📈 Success rate: {success_rate:.1f}%")
    
    if passed_checks == total_checks:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}✅ All checks passed!{Colors.END}")
        print(f"  {Colors.GREEN}The environment is properly set up.{Colors.END}\n")
        return 0
    else:
        print(f"\n  {Colors.YELLOW}{Colors.BOLD}⚠️  Some checks failed.{Colors.END}")
        print(f"  {Colors.YELLOW}Please fix the errors before continuing.{Colors.END}\n")
        return 1


def main():
    """Main entrypoint function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ROIXPLAIN ENVIRONMENT CHECK" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"{Colors.END}")
    
    all_results = {}
    
    # Main checks
    python_result = check_python_version()
    print_result(python_result)
    all_results["Python"] = [python_result]
    
    all_results["Packages"] = check_required_packages()
    all_results["Directories"] = check_directory_structure()
    all_results["Configurations"] = check_config_files()
    all_results["Data"] = check_data_files()
    all_results["Scripts"] = check_scripts()
    
    meridian_result = check_meridian_import()
    print_result(meridian_result)
    all_results["Meridian"] = [meridian_result]
    
    # Generate summary
    exit_code = generate_summary(all_results)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

