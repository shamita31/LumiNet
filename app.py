import streamlit as st
import torch
import torchvision
import cv2 as cv
import os
import time
import numpy as np
from PIL import Image

from objectDetection import detection_recognition
from lowlightenhancement import enhance_net_nopool


# Automatically use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lowlight(image_path):
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = np.asarray(data_lowlight, dtype=np.float32) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight)
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0).to(DEVICE)

    DCE_net = enhance_net_nopool().to(DEVICE)

    checkpoint = torch.load("./Epoch99.pth", map_location=DEVICE)
    DCE_net.load_state_dict(checkpoint)
    DCE_net.eval()

    start = time.time()

    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight)

    print(f"Enhancement time: {time.time() - start:.2f} seconds")

    torchvision.utils.save_image(enhanced_image.cpu(), "./test.png")


st.title("Low Light Image Enhancement and Object Detection")

st.header("Please upload an image")

file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file is not None:

    with torch.no_grad():
        img1 = Image.open(file)
        img1.save("./transit.png")
        lowlight("./transit.png")

    c = st.radio(
        label="Choose an option",
        options=["Denoising Image", "Without Denoising Image"]
    )

    img = cv.imread("test.png")

    if c == "Denoising Image":
        dst = cv.fastNlMeansDenoisingColored(
            img, None, 10, 10, 7, 21
        )

        image = Image.fromarray(dst)
        image.save("test_uni.png")

        image1 = detection_recognition("test_uni.png")

    else:
        image1 = detection_recognition("test.png")

    st.image(file, width=200, caption="Low Light Image")

    st.image(img, width=200, caption="Enhanced Image")

    st.image(
        cv.cvtColor(image1, cv.COLOR_BGR2RGB),
        width=200,
        caption="Object Detected Image"
    )