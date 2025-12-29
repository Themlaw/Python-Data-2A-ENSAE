# Python-Data-2A-ENSAE

**Authors:** Yann Gautier, Noé Idesheim, Titouan Constance

This is the GitHub repository for the Python project for Data Science in the second year at ENSAE.

This project studies the robustness of CAPTCHA recognition models against various types of noise. CAPTCHAs are a standard for web security, but their effectiveness can be compromised by image degradation.

**The central question of this project is as follows:**
> *How does image alteration via the injection of different noise types impact the resolution of CAPTCHAs by recognition models?*

We focus on:

1. The generation of a **synthetic CAPTCHA dataset** (based on MNIST).
2. The training of a **Multi-Head CNN** capable of recognizing 4 digits simultaneously.
3. The evaluation of **robustness** against two noise types (Gaussian, Salt & Pepper).
4. Comparison with a general-purpose **OCR API**.
5. Statistical analysis, including logistic regression on model performance.
6. Identification of the most challenging CAPTCHAs for security optimization.

The final model predicts the values of 4-digit CAPTCHAs under noise conditions.

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
4.  Get an API key:
  We use an API key to make requests to the https://ocr.space/ OCR API.
  To obtain your key, go to https://ocr.space/ocrapi/freekey and register. You will receive your API key by email.
  Then, add it to the file 'src/.env' as follows: 'OCR_API_KEY="your_api_key"'



## Code

- [MNIST.py](src/MNIST.py) : the class enabling us to load the MNIST data, apply noise, and generate our CAPTCHA database
- [simple_CNN.py](src/simple_CNN.py) : a simple CNN model for MNIST digit recognition
- [multi_head_CNN.py](src/multi_head_CNN.py) : the multi-head CNN model for recognizing 4-digit CAPTCHAs simultaneously
- [OCR_api.py](src/OCR_api.py) : functions for interacting with the OCR.space API for baseline comparison
- [evaluate_model.py](src/evaluate_model.py) : functions for evaluating model performance, robustness against noise, and statistical analysis

### Using the pipeline

[main.ipynb](src/main.ipynb)

## Used Data

Using a MNIST database, we generate a synthetic CAPTCHA dataset consisting of 4-digit sequences assembled from MNIST images, resized to 100x110 pixels. Noise (Gaussian or Salt & Pepper) can be dynamically applied at various intensities during training and evaluation.

[MNIST.py](src/MNIST.py)
