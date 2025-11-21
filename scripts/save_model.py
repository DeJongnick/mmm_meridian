import os
import hashlib
import pickle
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import pandas as pd

from run import build_model_and_sample, load_config_and_data, setup_seed, generate_html_report

# Get POC directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
POC_DIR = SCRIPT_DIR.parent.absolute()


def compute_data_hash(df: pd.DataFrame, config_data: dict) -> str:
    """
    Calcule un hash unique basé sur les données et la configuration du modèle.
    Cela permet d'identifier de manière unique un modèle par ses données d'entraînement.
    """
    # Créer une chaîne unique combinant :
    # - Le hash des données (premiers et derniers éléments pour performance)
    # - La configuration du modèle
    # - Les colonnes utilisées
    
    coord = config_data["coord_to_columns"]
    model_config = config_data["model_config"]
    
    # Informations sur les données
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
        # Hash des valeurs (échantillon pour performance)
        "data_hash": pd.util.hash_pandas_object(df).sum(),
    }
    
    # Configuration du modèle
    model_info = {
        "kpi_type": model_config["kpi_type"],
        "model_params": model_config.get("model", {}),
        "sampling": model_config.get("sampling", {}),
    }
    
    # Créer une chaîne combinée et calculer le hash
    combined_str = str(data_info) + str(model_info)
    data_hash = hashlib.sha256(combined_str.encode()).hexdigest()[:16]  # 16 caractères suffisent
    
    return data_hash


def save_model_with_report(config_file=None, data_file=None):
    """
    Entraîne un modèle, le sauvegarde dans outputs/models/{date_creation}/,
    et génère uniquement le report_data.html associé.
    """
    setup_seed()
    
    # Charger la configuration et les données
    config_data = load_config_and_data(config_file=config_file, data_file=data_file)
    df = config_data["df"]
    model_config = config_data["model_config"]
    
    # Créer un nom de dossier basé sur la date de création
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Calculer le hash des données pour les métadonnées (mais ne pas l'utiliser pour le nom du dossier)
    print("\n" + "="*60)
    print("📅 DATE DE CRÉATION")
    print("="*60)
    print(f"✓ Date: {date_folder}")
    data_hash = compute_data_hash(df, config_data)
    
    # Définir le chemin de sauvegarde
    output_base_dir = os.path.join(POC_DIR, "outputs", "models", date_folder)
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"\n📁 Dossier de sauvegarde: {output_base_dir}")
    
    # Construire et entraîner le modèle
    print("\n" + "="*60)
    print("🔨 CONSTRUCTION ET ENTRAÎNEMENT DU MODÈLE")
    print("="*60)
    mmm, model_config = build_model_and_sample(config_data)
    
    # Sauvegarder le modèle
    model_path = os.path.join(output_base_dir, "model.pkl")
    print(f"\n💾 Sauvegarde du modèle dans: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(mmm, f)
    print("✓ Modèle sauvegardé avec succès")
    
    # Sauvegarder les métadonnées
    now = datetime.now()
    metadata = {
        "folder_name": date_folder,
        "data_hash": data_hash,
        "created_at": now.isoformat(),
        "created_date": date_folder,  # Date du dossier pour faciliter la recherche
        "data_shape": df.shape,
        "data_columns": df.columns.tolist(),
        "date_range": {
            "start": str(df[config_data["coord_to_columns"].time].min()),
            "end": str(df[config_data["coord_to_columns"].time].max()),
        },
        "model_config": {
            "kpi_type": model_config["kpi_type"],
            "model_params": model_config.get("model", {}),
            "sampling": model_config.get("sampling", {}),
        },
        "config_file": config_file,
        "data_file": data_file,
    }
    
    metadata_path = os.path.join(output_base_dir, "metadata.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    print(f"✓ Métadonnées sauvegardées dans: {metadata_path}")
    
    # Générer uniquement le report_data.html
    print("\n" + "="*60)
    print("📄 GÉNÉRATION DU RAPPORT HTML")
    print("="*60)
    
    # Générer le rapport dans le dossier de sortie
    generate_html_report(mmm, model_config, output_dir=output_base_dir)
    
    print(f"\n✅ Modèle sauvegardé avec succès!")
    print(f"   📁 Emplacement: {output_base_dir}")
    print(f"   🤖 Modèle: model.pkl")
    print(f"   📄 Rapport: report_data.html")
    print(f"   📋 Métadonnées: metadata.yaml")
    
    return output_base_dir, date_folder


def list_saved_models():
    """Liste tous les modèles sauvegardés dans outputs/models/"""
    models_dir = os.path.join(POC_DIR, "outputs", "models")
    
    if not os.path.exists(models_dir):
        print("❌ Aucun modèle sauvegardé pour le moment.")
        return []
    
    models = []
    for model_folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_folder)
        if os.path.isdir(model_path):
            metadata_path = os.path.join(model_path, "metadata.yaml")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                models.append({
                    "folder": model_folder,
                    "path": model_path,
                    "metadata": metadata,
                })
            else:
                # Si pas de metadata, utiliser le nom du dossier comme date
                models.append({
                    "folder": model_folder,
                    "path": model_path,
                    "metadata": {"created_at": model_folder},
                })
    
    # Trier par date de création (plus récent en premier)
    return sorted(models, key=lambda x: x["folder"], reverse=True)


def display_saved_models():
    """Affiche la liste des modèles sauvegardés"""
    models = list_saved_models()
    
    if not models:
        print("❌ Aucun modèle sauvegardé pour le moment.")
        return
    
    print("\n" + "="*80)
    print("📚 MODÈLES SAUVEGARDÉS")
    print("="*80)
    
    for i, model_info in enumerate(models, 1):
        meta = model_info["metadata"]
        print(f"\n{i}. 📅 Date: {model_info['folder']}")
        created_at = meta.get('created_at', model_info['folder'])
        if created_at != model_info['folder']:
            print(f"   🕐 Créé le: {created_at}")
        if 'data_shape' in meta:
            print(f"   📊 Données: {meta.get('data_shape', 'N/A')}")
            print(f"   📅 Période: {meta.get('date_range', {}).get('start', 'N/A')} → {meta.get('date_range', {}).get('end', 'N/A')}")
        print(f"   📁 Chemin: {model_info['path']}")
        print(f"   📄 Rapport: {os.path.join(model_info['path'], 'report_data.html')}")
    
    print("\n" + "="*80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sauvegarde un modèle Meridian dans outputs/models/ avec report_data.html",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Utiliser le fichier de configuration par défaut
  python save_model.py
  
  # Spécifier un fichier de configuration
  python save_model.py --config config_v1.yaml
  
  # Spécifier un fichier de données
  python save_model.py --config config_v1.yaml --data data_processed.csv
  
  # Lister les modèles sauvegardés
  python save_model.py --list
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Fichier de configuration (ex: config_v1.yaml). Par défaut: config_v1.yaml"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Fichier de données CSV (ex: data_processed.csv). Par défaut: celui de la config"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Affiche la liste des modèles sauvegardés"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Afficher la liste si demandé
    if args.list:
        display_saved_models()
        exit(0)
    
    # Déterminer le fichier de configuration
    if args.config:
        config_file = f"configs/{args.config}"
        if not config_file.endswith('.yaml') and not config_file.endswith('.yml'):
            config_file += '.yaml'
    else:
        config_file = "configs/config_v1.yaml"
    
    # Vérifier que le fichier de config existe
    config_path = os.path.join(POC_DIR, config_file) if not os.path.isabs(config_file) else config_file
    if not os.path.exists(config_path):
        print(f"❌ Erreur: Le fichier de configuration '{config_file}' n'existe pas.")
        exit(1)
    
    print("\n" + "="*60)
    print("🚀 SAUVEGARDE DE MODÈLE MERIDIAN")
    print("="*60)
    print(f"📋 Configuration: {config_file}")
    if args.data:
        print(f"📊 Données: {args.data}")
    else:
        print(f"📊 Données: (celles de la configuration)")
    print("="*60)
    
    # Sauvegarder le modèle
    try:
        output_dir, date_folder = save_model_with_report(
            config_file=config_file,
            data_file=args.data
        )
    except Exception as e:
        print(f"\n❌ Erreur lors de la sauvegarde du modèle: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

