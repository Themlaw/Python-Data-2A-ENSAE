# Python-Data-2A-ENSAE

This is the GitHub repository for the Python project for Data Science in the second year at ENSAE.

This project explores an automatisation of the data preparation process.

First objective : the module detects the specificities of the dataset and applies the right transformations. For transparency, the module indicates the changes it made.

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
https://noise.visinf.tu-darmstadt.de/
https://arxiv.org/abs/1906.00270
