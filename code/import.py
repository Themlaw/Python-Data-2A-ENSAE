import kagglehub
import os
import shutil

def download_kaggle(dataset_name: str, target_directory: str, folders_to_copy: list) -> None:
    """
    Downloads a Kaggle dataset and copies specified folders to a target directory.

    Parameters:
    -----------
    dataset_name : str
      Name of the Kaggle dataset
    target_directory : str
      Directory where the dataset folders will be copied
    folders_to_copy : list
      List of folder names to copy from the dataset

    Returns:
    --------
    None

    """
    os.makedirs(target_directory, exist_ok=True)

    # Download the dataset
    dataset_path = kagglehub.dataset_download(dataset_name)

    for folder in folders_to_copy:
        source = os.path.join(dataset_path, folder)
        destination = os.path.join(target_directory, folder)

        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            print(f" Dossier introuvable : {source}")

if __name__ == "__main__":
    download_kaggle(
        dataset_name="rajneesh231/salt-and-pepper-noise-images",
        target_directory="input/salt_and_pepper_noise_images",
        folders_to_copy=["Ground_truth", "Noisy_folder"]
    )