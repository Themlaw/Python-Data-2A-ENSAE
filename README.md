# Python-Data-2A-ENSAE

This is the GitHub repository for the Python project for Data Science in the second year at ENSAE.

This project explores different methods of image classification and denoising in some specific training datasets.

First objective : Explore and build a classifier method limited to MNIST data

Final objective : Build a method to solve any Captcha

The final model should :

- predict the values of any Captcha characters

## Installation

1.  Install `uv`:

    ```shell
    pip install uv
    ```

2.  Create a virtual environment and download packages:

    ```shell
    uv sync
    ```

3.  Activate the virtual environment:

    -   On Windows:

        ```shell
        .venv\Scripts\activate
        ```

    -   On macOS/Linux:

        ```shell
        source .venv/bin/activate
        ```



## Code

- [MNIST.py](src/MNIST.py) : the class enabling us to load the MNIST data and make transforms on it, we also generate our database here
- [simple_CNN.py](src/simple_CNN.py) : the first model we use to predict the value of MNIST
- [evaluate_model.py](src/evaluate_model.py) : some functions used for making stats about the models

### Using the pipeline

[main.ipynb](src/main.ipynb)

## Used Data

Using a MNIST database, we generate a complete captcha database with images of 4 characters. Each image can have a varying level of noise on it.

[MNIST.py](src/MNIST.py)

## Questions

Denoise the data ? Optional, just see if the noise alters the quality of predictions
Generate the data ? Yes and it is equivalent to cleaning.
The website and scrapping ? No
Cleaning up the data ? Not necessary if data are generated
Getting the data from huggingface or from alternatives to kaggle ? Just for subdatasets, don't try it for the main dataset
Using the pipeline : is a bash command displayed in the readme ok ? + there are examples of usage ? No just make a main jupyter notebook with main + commented results.

Generate the data with different noise levels and make stats on the success rate according to the noise level. -> no

Décomposition en valeurs propres, couleurs, saturation. C'est rare de dénoiser puis de classifier, mais pas impossible. Classifier VAE, visualiser les espaces latents. Chaque région de l'espace correspond à une classe. Voir ça avec données sans noise, puis introduction du bruit et voir si le classifier est résistant, voir pourquoi l'accuracy baisse (classes plus segmentées ou plus vastes) Quelle classe est la plus robuste au noise ?
