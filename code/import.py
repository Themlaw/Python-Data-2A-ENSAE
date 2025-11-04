import os
import sys
import zipfile
import shutil
from pathlib import Path
import subprocess
import re

DATASET = "rajneesh231/salt-and-pepper-noise-images"
DEST_ROOT = Path("input/salt-and-pepper")
DEST_CLEAN = DEST_ROOT / "clean"
DEST_NOISY = DEST_ROOT / "noisy"
DEST_UNKNOWN = DEST_ROOT / "unknown"  # si un fichier n'est pas classable avec les règles ci-dessous

# mots-clés (insensibles à la casse) pour détecter le type dans le nom de dossier/fichier
CLEAN_TOKENS = ("clean", "pure", "original", "gt", "groundtruth")
NOISY_TOKENS = ("noisy", "noise", "salt", "pepper", "sp", "s&p")

def have_kaggle_credentials() -> bool:
    # 1) variables d'environnement ou 2) fichier ~/.kaggle/kaggle.json
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()

def pip_install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

def kaggle_download(dataset: str, out_dir: Path) -> Path:
    """
    Télécharge le dataset Kaggle en .zip dans out_dir et renvoie le chemin du zip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Installe le client kaggle si nécessaire
    try:
        import kaggle  # noqa: F401
    except ImportError:
        pip_install("kaggle")

    zip_path = out_dir / (dataset.split("/")[-1] + ".zip")
    if zip_path.exists():
        return zip_path

    # Appelle la CLI kaggle (plus simple que l'API Python pour les datasets publics)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset, "-p", str(out_dir), "-f", "",  # -f vide -> tout
        "--force"
    ]
    # La CLI écrit <slug>.zip dans out_dir
    subprocess.check_call(cmd)
    # Le nom réel du fichier peut varier; on récupère le premier .zip déposé
    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError("Aucun fichier .zip téléchargé par la CLI kaggle.")
    return zips[0]

def safe_extract(zip_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # extraire en évitant la traversal
        for member in zf.infolist():
            member_path = extract_to / member.filename
            if not str(member_path.resolve()).startswith(str(extract_to.resolve())):
                continue
            zf.extract(member, extract_to)

def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

def classify_from_name(p: Path) -> str:
    """
    Classe "clean" / "noisy" à partir des noms de fichier et de dossier.
    Retourne "clean", "noisy" ou "unknown".
    """
    haystack = "/".join([*p.parts]).lower()
    # Nettoyage des caractères non alpha pour des matchs plus robustes
    tokens = re.split(r"[^a-z0-9]+", haystack)

    def contains_any(keys):
        return any(k in tokens or k in haystack for k in keys)

    if contains_any(NOISY_TOKENS) and not contains_any(CLEAN_TOKENS):
        return "noisy"
    if contains_any(CLEAN_TOKENS) and not contains_any(NOISY_TOKENS):
        return "clean"
    # Heuristique : si deux versions existent côte à côte, on tranchera au moment du déplacement par appariement optionnel.
    return "unknown"

def move_with_structure(src_file: Path):
    cls = classify_from_name(src_file)
    if cls == "clean":
        dest_dir = DEST_CLEAN
    elif cls == "noisy":
        dest_dir = DEST_NOISY
    else:
        dest_dir = DEST_UNKNOWN
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Conserver un sous-arbre minimal si nécessaire pour éviter collisions
    dest_file = dest_dir / src_file.name
    # si collision, suffixer
    if dest_file.exists():
        stem, suf = dest_file.stem, dest_file.suffix
        i = 2
        while dest_file.exists():
            dest_file = dest_dir / f"{stem}_{i}{suf}"
            i += 1
    shutil.move(str(src_file), str(dest_file))

def main():
    if not have_kaggle_credentials():
        raise RuntimeError(
            "Identifiants Kaggle manquants. "
            "Placez votre 'kaggle.json' dans ~/.kaggle/ (chmod 600) "
            "ou définissez KAGGLE_USERNAME et KAGGLE_KEY."
        )

    # 1) Télécharger
    download_dir = Path("downloads")
    zip_path = kaggle_download(DATASET, download_dir)

    # 2) Extraire dans un répertoire temporaire
    tmp_extract = Path("downloads/_extract_salt_pepper")
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    safe_extract(zip_path, tmp_extract)

    # 3) Créer la destination propre
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    DEST_CLEAN.mkdir(exist_ok=True, parents=True)
    DEST_NOISY.mkdir(exist_ok=True, parents=True)

    # 4) Parcourir les images extraites et les classer
    files = [p for p in tmp_extract.rglob("*") if p.is_file() and is_image(p)]
    if not files:
        # Si certains datasets mettent tout dans un zip interne, tenter de dézipper les .zip imbriqués
        inner_zips = list(tmp_extract.rglob("*.zip"))
        for z in inner_zips:
            safe_extract(z, z.parent)
        files = [p for p in tmp_extract.rglob("*") if p.is_file() and is_image(p)]

    if not files:
        raise RuntimeError("Aucune image trouvée après extraction. Le contenu du zip est inhabituel.")

    for f in files:
        move_with_structure(f)

    # 5) (Optionnel) petit appariement pour fichiers 'unknown'
    #    Si un même nom existe en double (ex: foo.png et foo_noisy.png), tenter de déduire.
    unknowns = list(DEST_UNKNOWN.glob("*"))
    for u in unknowns:
        name = u.stem.lower()
        if any(k in name for k in NOISY_TOKENS):
            shutil.move(str(u), str(DEST_NOISY / u.name))
        elif any(k in name for k in CLEAN_TOKENS):
            shutil.move(str(u), str(DEST_CLEAN / u.name))

    # 6) Nettoyage
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)

    print(f"Téléchargé depuis Kaggle: {zip_path}")
    print(f"Images 'clean' -> {DEST_CLEAN.resolve()}")
    print(f"Images 'noisy' -> {DEST_NOISY.resolve()}")
    unknowns = list(DEST_UNKNOWN.glob("*"))
    if unknowns:
        print(f"Non classés -> {DEST_UNKNOWN.resolve()} (à vérifier manuellement: {len(unknowns)} fichiers)")
    else:
        if DEST_UNKNOWN.exists():
            DEST_UNKNOWN.rmdir()

if __name__ == "__main__":
    main()
