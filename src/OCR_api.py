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
    
    # Conversion to uint8 if necessary
    if image_data.max() <= 1.0:
        image_data = (image_data * 255).astype(np.uint8)
    else:
        image_data = image_data.astype(np.uint8)  
    
    # Remove singleton channel dimension if present (e.g., shape (H, W, 1) -> (H, W))
    if image_data.ndim == 3 and image_data.shape[-1] == 1:
        image_data = image_data.squeeze(axis=-1)
        
    # Image creation and processing
    img = Image.fromarray(image_data)
        
    # Convert to grayscale if necessary
    if img.mode != 'L':
        img = img.convert('L')
        
    # Image color inversion
    img = ImageOps.invert(img)
       
    # Sharpness enhancement
    img = img.filter(ImageFilter.SHARPEN)
        
    # Temporary file saving
    temp_filename = "test1.png"
    img.save(temp_filename)
        
    # Prediction via API
    print(f"Predicting image...")
    test_file = ocr_space_file(
        filename=temp_filename, 
        api_key=api_key, 
        language=language, 
        filetype=filetype, 
        OCREngine=OCREngine
    )
        
    # Parse result into dictionary
    data = json.loads(test_file)
    # Access parsed 
    try:
        parsed_text = data["ParsedResults"][0]["ParsedText"]
        return parsed_text
    except:
        print(f"Error: {data}")
        return np.nan        
 
def test_ocr_on_captcha(mnist_loader, api_key, h5_filepath, num_images=3, noise_factor=0.1, display_images=True):
    """
    Tests the OCR API on CAPTCHA images generated from the MNIST dataset.
    
    :param mnist_loader: Instance of MnistDataloader
    :param api_key: API Key for OCR.space
    :param h5_filepath: Path to the HDF5 file containing the CAPTCHAs
    :param num_images: Number of images to test
    :param noise_factor: Noise factor to apply to the images
    :param display_images: If True, displays the images using matplotlib
    :return: List of results (index, predicted text, ground truth text)
    """
    # CAPTCHA generation
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
    
    # Testing API on X_train images
    for i in range(len(X_train)):
        image_data = X_train[i]
        
        # Conversion to uint8 if necessary
        if image_data.max() <= 1.0:
            image_data = (image_data * 255).astype(np.uint8)
        else:
            image_data = image_data.astype(np.uint8)
        
        # Image creation and processing
        img = Image.fromarray(image_data)
        
        # Convert to grayscale if necessary
        if img.mode != 'L':
            img = img.convert('L')
        
        # Image color inversion
        img = ImageOps.invert(img)
        
        # Sharpness enhancement
        img = img.filter(ImageFilter.SHARPEN)
        
        # Temporary file saving
        temp_filename = "test1.png"
        img.save(temp_filename)
        
        # Prediction via API
        print(f"Predicting image {i}...")
        test_file = ocr_space_file(
            filename=temp_filename, 
            api_key=api_key, 
            language='eng', 
            filetype='PNG', 
            OCREngine=2
        )
        
        # Parse result into dictionary
        data = json.loads(test_file)
        
        # Access parsed text
        parsed_text = data["ParsedResults"][0]["ParsedText"]
        
        # Retrieving ground truth label
        true_label = ''.join(map(str, y_train[i]))
                
        results.append({
            'index': i,
            'predicted': parsed_text,
            'true_label': true_label,
            'correct': parsed_text.strip() == true_label
        })
        
        # Displaying image
        if display_images:
            plt.figure(figsize=(8, 4))
            plt.imshow(img, cmap='gray')
            plt.title(f"Image {i}: Predicted = '{parsed_text}', Ground Truth = '{true_label}'")
            plt.axis('off')
            plt.show()
    
    # Cleanup temporary file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return results


if __name__ == '__main__':
    from MNIST import MnistDataloader
    
    # Loading API key
    load_dotenv(dotenv_path="src/.env") 
    api_key = os.getenv("OCR_API_KEY")
    
    if api_key is None:
        print("Error: API Key not found. Check your .env file")
    else:
        # MNIST loader initialization
        mnist = MnistDataloader()
        
        # Path to CAPTCHA file
        h5_filepath = os.path.join('data', 'captcha_data', 'captcha_dataset.h5')
        
        # Testing OCR on CAPTCHAs
        results = test_ocr_on_captcha(
            mnist_loader=mnist,
            api_key=api_key,
            h5_filepath=h5_filepath,
            num_images=3,
            noise_factor=0.1,
            display_images=True
        )
        
        # Displaying summary
        for r in results:
            print(f"Image {r['index']}: Predicted = '{r['predicted']}', Ground Truth = '{r['true_label']}'")