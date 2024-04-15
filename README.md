## Asterix 
<img src="images/logo.png" alt="Logo" width="200">     

### What is this about?   
This is the prototyping repo for the Asterix project. We develop and showcase prototype methods for compressing VDFs in Vlasiator.
So far this has been happening in our shared jupyter notebook ```Inno4Scale.ipynb```.     
Currently we utilize the following methods:
+ Zlib lossless compression.
+ ZFP lossy compression.
+ Using a Multi Layer Perceptron [MLP] to train on and compress VDFs.
+ Using a Multi Layer Perceptron as above enriched with Fourier Features.
+ Using a Convolutional Neural Network.
+ Using a Gaussian Mixture Model.
+ Using a Hermite Decomposition.
+ Using a Spherical Harmonic Decomposition.

### Requirements
+ Cargo [instructions](https://doc.rust-lang.org/cargo/getting-started/installation.html)   
+ Maturin [GitHub repo](https://github.com/PyO3/maturin)   
+ pyzfp [instructions](https://pypi.org/project/pyzfp/)   
+ zlib (distro-dependent)   
+ Python >3.7   
+ Pytorch [instructions](https://pytorch.org/get-started/locally/#linux-installation)
+ Analysator [GitHub Repo](https://github.com/fmihpc/analysator)    

### Running the Project
The MLP used, is hosted in this repo and is written in Rust. To use it from python we just need to build a shared library that wraps over it. Thankfully this is not too hard to do!    
+ Do a ```pip install -r requirements.txt``` to get all the needed packages.    
+ Inside your virtual environment you can build the project with ```maturin develop --release```.      
+ Open up the notebook ```Inno4Scale.ipynb``` and have fun.   
+ With each fresh commit we try to maintain a PDF form of the notebook under pdf/.
+ On a modern system with an NVIDIA GPU running the full notebook should take less than 3 minutes.   

