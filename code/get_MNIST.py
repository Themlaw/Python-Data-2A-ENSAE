import kagglehub
import shutil
import os

# Configuration
TARGET_DIR = os.path.join(".", "ressources", "MNIST")

# Télécharger le dataset (dans le cache par défaut)
print("Téléchargement du dataset MNIST...")
cache_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print(f"Dataset téléchargé dans le cache : {cache_path}")

# Créer le dossier cible s'il n'existe pas
os.makedirs(TARGET_DIR, exist_ok=True)

# Copier les fichiers vers le dossier cible
print(f"\nCopie vers {TARGET_DIR}...")
for item in os.listdir(cache_path):
    source = os.path.join(cache_path, item)
    destination = os.path.join(TARGET_DIR, item)
    
    if os.path.isfile(source):
        shutil.copy2(source, destination)
        print(f"  {item}")
    elif os.path.isdir(source):
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print(f"  {item}/ (dossier)")

print(f"\nDataset MNIST disponible dans : {os.path.abspath(TARGET_DIR)}")