from numpy.lib.index_tricks import unravel_index
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
from torch.functional import Tensor
from model_Tom import CNN
import torch
import torchvision

#Load state of cnn (trained) (public load)
cnn = CNN()
cnn.load_state_dict(torch.load('../data/model.pth'))
cnn.eval()
##
def main():
    
    st.title("CNN Number Recognition")
    # model = load_model("../training_model/model.npy")
    canvas_result = display_canvas()
    if st.sidebar.button("Guess the number!"):
        converted_canvas_img_28x28 = convert_Canvas(canvas_result)
        forward_propagate(converted_canvas_img_28x28)

def draw_number(img: np.ndarray, label: str, imgsize: int) -> None:
    # Thanks to Andres Weber for the canvas Layout
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    st.subheader(f"First Number: {label}")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width=imgsize, use_column_width=imgsize)

def forward_propagate(img:Tensor):
    #[batch_size, channels, height, width] --> Tensor Should look like this 
    # forward through neural network.
    output=cnn(img) 
    _,pred=torch.max(output,1)
    # Print predict
    st.write("Predicted digit is : " + str(pred.item()))    
    # training_img = sc.fit_transform(training_img)
    print(pred)
    #st.subheader("Number Prediction")
    #st.write("Predicted digit is : " + str(predicted_digit))


def display_canvas():
    # Thanks to Andres Weber for the canvas Layout
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 25, 50, 45)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#ffffff")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
    canvas_size = 560 
    drawing_mode = "freedraw"
    # realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    st.subheader("Draw A Number and I'll guess it!")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=True,
        height=canvas_size,
        width=canvas_size,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    return canvas_result

def convert_Canvas(canvas) -> Tensor:
    # convert rgba img to gray-scale img
    gray_img = np.dot(canvas.image_data[...,:3], [0.299, 0.587, 0.114])
    #tensor = torch.from_numpy(gray_img)
    # define type for Float
    #tensor=tensor.type('torch.FloatTensor')
    # Add dimensions for [1,1,28,28] shape
    #tensor = tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    #define transformer

    # Print Out max value of array
    print(gray_img.shape)
    x,y=unravel_index(gray_img.argmax(), gray_img.shape)
    print (gray_img[x,y])
    ###
    #normalize img because torchvision.transforms.ToTensor() does not (just on Pillow Images) --> https://pytorch.org/vision/stable/transforms.html just transformes when numpy array 
    # has form (x,y,c) 3 dimesnions with channel dimension ours has (x,y)
    gray_img=gray_img*1/255
    # Transform to Tensor resize to 28x28 and normalize with std and mean from dataset
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=[28,28]), 
        torchvision.transforms.Normalize(
                                 (0.13195264339447021,), (0.30903369188308716,))
                             ])
    # Use Transformer on numpy array and add an dimension for [1, 1, 28, 28] format
    tensor_img=transform(gray_img).unsqueeze(dim=0)
    # Cast Type from torch.float64 to torch.float32
    tensor_img=tensor_img.type('torch.FloatTensor')
    #print(tensor_img.dtype)

    # Print out max of transformed array
    #print(tensor_img.data[0].numpy())
    #transformed_img=tensor_img.data[0].numpy()
    #print(transformed_img.shape)
    #x1,y1=unravel_index(transformed_img.argmax(), transformed_img.shape)
    #print(transformed_img[x1,y1])
    # Print out distribution (normal-distribution) (after normalize Transformer)
    print(tensor_img.shape) #shape
    mean=tensor_img.data.float().mean()
    std=tensor_img.data.float().std()
    print(f"Mean: {mean}") # mean
    print(f"Std: {std}")  # std
    plt.imshow(tensor_img.data[0][0].numpy())
    draw_number(tensor_img.data[0][0].numpy(), "converted image from canvas and normalized with tensor", 392)
    return tensor_img


if __name__ == "__main__":
    main()