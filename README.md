
## Install miniconda for virtual environment management for Windows or Mac:**
https://www.anaconda.com/download/success

Make sure you enable this option during the installation:


## **01-Prepare a conda environment for the app:**
## Install cuda 12.6 (only for Windows uers):**

- **create an env**: 
	- conda create --name generative_ai python==3.9

- **check if it's installed:** 
	- conda env list 

- **activate the environment:** 
	- conda activate generative_ai

- **Install the required libraries:**
	- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
	- pip install diffusers
	- pip install accelerate
	- pip install transformers 
	- pip install opencv-python
    - pip install pyOpenSSL
    - pip install controlnet_aux
    - pip install pillow
    - pip install mediapipe
    - pip install numpy
    - pip install timm
    - pip install peft
	- One line to download all the libraries: 
    - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && pip install diffusers accelerate transformers opencv-python pyOpenSSL controlnet_aux pillow mediapipe numpy timm peft