import requests
import os
from dotenv import load_dotenv
import json
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

def ocr_space_file(filename, overlay=False, api_key='helloworld', language = 'eng', filetype = 'PNG', OCREngine = 2):
    """ OCR.space API request with local file.
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               'filetype': filetype,
               'OCREngine': OCREngine,
            #    'Scale': True
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload, 
                          )
    return r.content.decode()

def ocr_space_one_image(image_data, api_key, language='eng', filetype='PNG', OCREngine=2):
    """ OCR.space API request with one image in matrix format.
    :param image_data: Image in matrix format.
    :param api_key: OCR.space API key.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """       
    
    # Conversion en uint8 si nécessaire
    if image_data.max() <= 1.0:
        image_data = (image_data * 255).astype(np.uint8)
    else:
        image_data = image_data.astype(np.uint8)  
        
    # Création et traitement de l'image
    img = Image.fromarray(image_data)
        
    # Conversion en niveaux de gris si nécessaire
    if img.mode != 'L':
        img = img.convert('L')
        
    # Inversion des couleurs de l'image
    img = ImageOps.invert(img)
       
    # Amélioration de la netteté
    img = img.filter(ImageFilter.SHARPEN)
        
    # Sauvegarde temporaire
    temp_filename = "test1.png"
    img.save(temp_filename)
        
    # Prédiction via l'API
    print(f"Prédiction de l'image...")
    test_file = ocr_space_file(
        filename=temp_filename, 
        api_key=api_key, 
        language=language, 
        filetype=filetype, 
        OCREngine=OCREngine
    )
        
    # Conversion du résultat en dictionnaire
    data = json.loads(test_file)
    print(data)
    # Accès au texte parsé
    parsed_text = data["ParsedResults"][0]["ParsedText"]
        
    return parsed_text
 
def test_ocr_on_captcha(mnist_loader, api_key, h5_filepath, num_images=3, noise_factor=0.1, display_images=True):
    """
    Teste l'API OCR sur des images CAPTCHA générées à partir du dataset MNIST.
    
    :param mnist_loader: Instance de MnistDataloader
    :param api_key: Clé API pour OCR.space
    :param h5_filepath: Chemin vers le fichier HDF5 contenant les CAPTCHA
    :param num_images: Nombre d'images à tester
    :param noise_factor: Facteur de bruit à appliquer aux images
    :param display_images: Si True, affiche les images avec matplotlib
    :return: Liste des résultats (index, texte prédit, texte réel)
    """
    # Génération des CAPTCHA
    x = mnist_loader.load_captcha_dataset(
        h5_filepath=h5_filepath, 
        num_images_train=num_images, 
        num_images_test=0, 
        random_selection=False, 
        apply_noise=True, 
        noise_type='gaussian', 
        noise_factor=noise_factor, 
        rgb_noise=False
    )
    
    (X_train, y_train), (X_test, y_test) = x
    
    results = []
    
    # Test de l'API sur les images X_train
    for i in range(len(X_train)):
        image_data = X_train[i]
        
        # Conversion en uint8 si nécessaire
        if image_data.max() <= 1.0:
            image_data = (image_data * 255).astype(np.uint8)
        else:
            image_data = image_data.astype(np.uint8)
        
        # Création et traitement de l'image
        img = Image.fromarray(image_data)
        
        # Conversion en niveaux de gris si nécessaire
        if img.mode != 'L':
            img = img.convert('L')
        
        # Inversion des couleurs de l'image
        img = ImageOps.invert(img)
        
        # Amélioration de la netteté
        img = img.filter(ImageFilter.SHARPEN)
        
        # Sauvegarde temporaire
        temp_filename = "test1.png"
        img.save(temp_filename)
        
        # Prédiction via l'API
        print(f"Prédiction de l'image {i}...")
        test_file = ocr_space_file(
            filename=temp_filename, 
            api_key=api_key, 
            language='eng', 
            filetype='PNG', 
            OCREngine=2
        )
        
        # Conversion du résultat en dictionnaire
        data = json.loads(test_file)
        
        # Accès au texte parsé
        parsed_text = data["ParsedResults"][0]["ParsedText"]
        
        # Récupération du label réel
        true_label = ''.join(map(str, y_train[i]))
                
        results.append({
            'index': i,
            'predicted': parsed_text,
            'true_label': true_label,
            'correct': parsed_text.strip() == true_label
        })
        
        # Affichage de l'image
        if display_images:
            plt.figure(figsize=(8, 4))
            plt.imshow(img, cmap='gray')
            plt.title(f"Image {i}: Prédit = '{parsed_text}', Réel = '{true_label}'")
            plt.axis('off')
            plt.show()
    
    # Nettoyage du fichier temporaire
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return results


if __name__ == '__main__':
    from MNIST import MnistDataloader
    
    # Chargement de la clé API
    load_dotenv(dotenv_path="src/.env") 
    api_key = os.getenv("OCR_API_KEY")
    
    if api_key is None:
        print("Erreur: Clé API non trouvée. Vérifiez votre fichier .env")
    else:
        # Initialisation du loader MNIST
        mnist = MnistDataloader()
        
        # Chemin vers le fichier CAPTCHA
        h5_filepath = os.path.join('data', 'captcha_data', 'captcha_dataset.h5')
        
        # Test de l'OCR sur les CAPTCHA
        results = test_ocr_on_captcha(
            mnist_loader=mnist,
            api_key=api_key,
            h5_filepath=h5_filepath,
            num_images=3,
            noise_factor=0.1,
            display_images=True
        )
        
        # Affichage du résumé
        for r in results:
            print(f"Image {r['index']}: Prédit = '{r['predicted']}', Réel = '{r['true_label']}'")
        