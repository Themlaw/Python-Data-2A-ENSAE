# Python-Data-2A-ENSAE

This is the GitHub repository for the Python project for Data Science in the second year at ENSAE.

This project explores differents methods of image denoising in somes trainigs specifics datasets.

First objective : Explore and build a method limited to black and white images.

## Code

- [pipeline.py](code/pipeline.py) : the pipeline with all parameters
- [main.py](code/main.py) : main file + parser

### Requirements

- Python
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

https://www.kaggle.com/datasets/rajneesh231/salt-and-pepper-noise-images

https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset

https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset
