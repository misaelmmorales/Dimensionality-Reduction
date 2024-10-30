# Advanced Dimensionality Reduction
***
Simple workflows for advanced dimensionality reduction techniques. 
- [Principal Component Analsysis](https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/ADR_PCA.ipynb)
- [Singular Value Decomposition](https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/ADR_SVD.ipynb)
- [Dictionary Learning](https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/ADR_DictionaryLearning.ipynb)
- [Manifold Learning](https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/ADR_Manifold.ipynb)
- [Fourier Transform / Wavelet Transform](https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/ADR_DFT_DWT.ipynb)
- [Dynamic Mode Decomposition](https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/ADR_POD_DMD.ipynb)
- [Deep Learning AutoEncoders](https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/ADR_DeepLearn.ipynb)

***
Each workflow demonstrates a different dimensionality reduction technique. For demonstration purposes, we use the MNIST (handwritten digits) dataset. This dataset contains 60,000 instances of 28x28 images of digits 0 through 9.
<p align="center">
  <img src="https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/images/MNIST_samples.png", width=400>
</p> 

The main goal of each workflow is simple. (1) Encode the MNIST dataset into a latent variable of reduced dimension, (2) reconstruct the image from this reduced latent space, and for experimental purposes (3) generate a reconstruction of a subsurface property from the MNIST latent represenation previously encoded using the given dimensionality reduction technique. For this, we can have a sparse subsurface sample, and interpolate the property of interest.
<p align="center">
  <img src="https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/images/enc_dec_diagram.png", width=400>
</p> 
<p align="center">
    <img src="https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/images/interp_subsurface_data.png", width=500>
</p>

***
For each dimensionality reduction technique, we will obtain a basis (tailored/generic), a latent space representation, and a reconstructed image. 

For example, with SVD we obtain:
<p align="center">
  <img src="https://github.com/misaelmmorales/Dimensionality-Reduction/blob/main/images/adr_complete.JPG">
</p> 
where MNIST images are projected onto 36 latent variables and then reconstructed, and where we compare the results for different latent space dimensions for a subsurface transfer learning reconstruction.

***
Remarks:
- The MNIST dataset has the digits centered at the image. Therefore, we using the learned basis from this generic dataset, our subsurface image reconstructions will be most accurate near the center and quite bad near the edges. If we increase the dimension of the latent space, we start to gather information of the edges (which is irrelevant for MNIST but relevant for subsurface maps). Therefore, for this particular transfer learning problem, we might need to use more latent variables than desired. Using a different generic dataset with more information near the edges to learn the sparse encoding might help in transfer reconstruction.
- For MNIST compression and reconstruction, our experiment shows that 36 latent variables are sufficient to obtain a very good MNIST reconstruction, with approximately MSE=0.01 and SSIM=0.85!
- For subsurface compression and reconstruction, we tend to need a larger latent manifold in order to find the information necessary for good reconstruction. However, we can still see from simple experiments that using medium-sized latent spaces we can obtain MSE's of about 0.15 and SSIM's of about 0.6!

  ***
  contact: github.com/misaelmmorales
