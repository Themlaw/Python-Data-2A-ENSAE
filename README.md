# Python-Data-2A-ENSAE

This is the GitHub repository for the Python project for Data Science in the second year at ENSAE.

This project explores different methods of image denoising in some specific training datasets.

First objective : Explore and build a classifier method limited to MNIST data

Final objective : Build a method to solve any Captcha

The final model should :

- denoise captcha images
- separate the characters
- predict the value of the character

## Code

- [pipeline.py](code/pipeline.py) : the pipeline with all parameters
- [main.py](code/main.py) : main file + parser

### Requirements

- python
- numpy
- pandas
- sktime
- scikit-learn
- matplotlib

### Using the pipeline

[main.py](code/main.py)

```bash
Arguments:
-p --path        : Path to the file containing the dataset

Examples:
> python main.py --path ./input/path
```

## Used Data

We create a website, url :

This website stores our generated data with 3 variables for each line : image, solution, noise

## Questions

Denoise the data ? Optional, just see if the noise alters the quality of predictions
Generate the data ? Yes and it is equivalent to cleaning.
The website and scrapping ? No
Cleaning up the data ? Not necessary if data are generated
Getting the data from huggingface or from alternatives to kaggle ? Just for subdatasets, don't try it for the main dataset
Using the pipeline : is a bash command displayed in the readme ok ? + there are examples of usage ? No just make a main jupyter notebook with main + commented results.

Generate the data with different noise levels and make stats on the success rate according to the noise level. -> no

Décomposition en valeurs propres, couleurs, saturation. C'est rare de dénoiser puis de classifier, mais pas impossible. Classifier VAE, visualiser les espaces latents. Chaque région de l'espace correspond à une classe. Voir ça avec donnes sans noise, puis introduction du bruit et si le classifier est résistant, voir pourquoi l'accuracy baisse (classes plus segmentées ou plus vastes) Quelle classe est la plus robuste au noise ?