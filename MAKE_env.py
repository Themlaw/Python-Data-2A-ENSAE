import subprocess
import sys
import os
import platform

# Configuration
VENV_NAME = ".DATA_env"
REQUIREMENTS_FILE = "requirements.txt"

def run_command(command, description):
    """Exécute une commande système avec gestion d'erreur."""
    print(f"--- {description} ---")
    try:
        subprocess.run(command, check=True)
        print("✅ Succès.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de : {description}")
        print(f"Détail : {e}")
        sys.exit(1)

def get_venv_python():
    """Retourne le chemin vers le Python de l'environnement virtuel."""
    if platform.system() == "Windows":
        return os.path.join(VENV_NAME, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_NAME, "bin", "python")

def main():
    print(f"Démarrage de l'installation du projet...\n")

    # 1. Créer l'environnement virtuel (avec le Python système)
    if not os.path.exists(VENV_NAME):
        cmd = [sys.executable, "-m", "venv", VENV_NAME]
        run_command(cmd, f"Création du venv '{VENV_NAME}'")
    else:
        print(f"ℹ️ Le dossier {VENV_NAME} existe déjà. On continue.\n")

    # 2. Obtenir le chemin du Python de l'env
    venv_python = get_venv_python()
    
    # 3. Installer/Mettre à jour pip dans l'env
    print("⚙️ Mise à jour de pip dans l'environnement virtuel...")
    run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"], 
                "Mise à jour de pip")

    # 4. Installer uv dans l'env
    print("⚙️ Installation de uv dans l'environnement virtuel...")
    run_command([venv_python, "-m", "pip", "install", "uv"], 
                "Installation de uv")

    # 5. Installer les dépendances si requirements.txt existe
    if os.path.exists(REQUIREMENTS_FILE):
        cmd = [venv_python, "-m", "uv", "pip", "install", "-r", REQUIREMENTS_FILE]
        run_command(cmd, "Installation des modules (via uv)")
    else:
        print(f"ℹ️ Aucun fichier {REQUIREMENTS_FILE} trouvé. Pas de modules à installer.\n")

    # 6. Afficher comment activer
    print("Installation terminée !")
    print("Pour activer l'environnement, lance cette commande :\n")
    
    if platform.system() == "Windows":
        print(f"    {VENV_NAME}\\Scripts\\activate")
    else:
        print(f"    source {VENV_NAME}/bin/activate")

if __name__ == "__main__":
    main()