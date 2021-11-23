# CNN-Numbers

## Installation

Use Pipenv:

```pipenv install```

Use pip:

```pip install -r requirements.txt```

## Jupyter Notebook


  A jupyter notebook for training a convolutional neural network (CNN) on the MNIST dataset
 
  * Accuaracy: 99% on random test data from the MNIST dataset 
  * A visualisation of the Loss-Function over the training process
  * Testplots
  * Autodownloader and split for the training and validation data
  
  Modell-Architecture:
  
  ![Image CNN](img/CNN_Architecture.png)
 
  [Image Source](https://ravivaishnav20.medium.com/handwritten-digit-recognition-using-pytorch-get-99-5-accuracy-in-20-k-parameters-bcb0a2bdfa09)
  
  Framerwork: [pytorch](https://pytorch.org)
  
## Web Test-Application


  A streamlit Application for Testing the CNN
  
  Start the Application inside the `app` directory with:
  
  
```streamlit run app.py```  

 
  Draw your own numbers inside a canvas and let the CNN guess
  
  Framerwork: [streamlit](https://streamlit.io)
  
*Thanks to Andreas Weber for the canvas-idea*
