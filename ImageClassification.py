# import dependencies
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18



# load model

# @st.cache_data
def load_model():
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 2),
        torch.nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model=load_model()

# The app
st.title("Pneumonia Detection from X-ray Images")

uploaded_file = st.file_uploader("Upload an X-ray image", type="jpg")
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # converting images to RGB
    class ConvertToRGB():
        def __call__(self , img):
            if img.mode != 'RGB':
                img = img.convert("RGB")
            return img


    # normalizing the data using transforms

    height = 224
    width = 224

    transform_norm = transforms.Compose(

        [
            ConvertToRGB(),
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

    image_tensor = transform_norm(image).unsqueeze(0)

    # Prediction

    outputs = model(image_tensor)
    _, pred = torch.max(outputs,1)
    label = "Pneumonia" if pred.item() == 1 else "Normal"

    # displaying results
    st.image(image, caption=f"Prediction: {label}", use_container_width=True)


    # visualize
    plt.figure(figsize=(9, 9))
    image = np.array(image)
    plt.show()
    plt.imshow(image, cmap='jet', alpha=0.5)
    plt.axis('off')
    st.pyplot(plt)







