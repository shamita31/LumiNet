import streamlit as st
import torch
import torchvision
import cv2 as cv
import time
import numpy as np
from PIL import Image

import objectDetection
from lowlightenhancement import enhance_net_nopool


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LumiNet - Low Light Enhancement",
    page_icon="🌙",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌙 LumiNet")
st.subheader(
    "Low Light Image Enhancement and Object Detection"
)

st.write(
    "Enhance low-light images using Zero-DCE and "
    "detect objects using YOLOv3."
)


# ============================================================
# SIDEBAR - PARAMETERS
# ============================================================

st.sidebar.header("⚙️ Parameters")

# ------------------------------------------------------------
# Object Detection
# ------------------------------------------------------------

st.sidebar.subheader("🎯 Object Detection")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.20,
    step=0.05,
    help="Minimum confidence required for object detection."
)

iou_threshold = st.sidebar.slider(
    "IoU / NMS Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.20,
    step=0.05,
    help="Threshold used for Non-Maximum Suppression."
)


# ------------------------------------------------------------
# Image Processing
# ------------------------------------------------------------

st.sidebar.subheader("🖼️ Image Processing")

denoise_strength = st.sidebar.slider(
    "Denoising Strength",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="Strength of OpenCV denoising."
)

edge_style = st.sidebar.selectbox(
    "Edge Style",
    [
        "None",
        "Canny",
        "Sobel",
        "Laplacian"
    ]
)


# ------------------------------------------------------------
# Depth
# ------------------------------------------------------------

st.sidebar.subheader("🗺️ Depth Estimator")

depth_estimator = st.sidebar.selectbox(
    "Depth Estimator",
    [
        "Disabled",
        "Pseudo Depth (Brightness)"
    ],
    help=(
        "The current project does not contain a trained "
        "depth-estimation model. Pseudo Depth is only "
        "a brightness-based visualization."
    )
)


# ------------------------------------------------------------
# System
# ------------------------------------------------------------

st.sidebar.subheader("💻 System")

st.sidebar.info(
    "Device: " + str(DEVICE).upper() +
    "\n\nPyTorch: " + torch.__version__
)


# ============================================================
# ZERO-DCE MODEL
# ============================================================

@st.cache_resource
def load_enhancement_model():

    model = enhance_net_nopool().to(DEVICE)

    checkpoint = torch.load(
        "./Epoch99.pth",
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model


# ============================================================
# LOW LIGHT ENHANCEMENT
# ============================================================

def lowlight(image):

    data_lowlight = image.convert("RGB")

    data_lowlight = np.asarray(
        data_lowlight,
        dtype=np.float32
    ) / 255.0

    data_lowlight = torch.from_numpy(
        data_lowlight
    )

    data_lowlight = data_lowlight.permute(
        2,
        0,
        1
    )

    data_lowlight = data_lowlight.unsqueeze(
        0
    ).to(DEVICE)

    DCE_net = load_enhancement_model()

    start = time.time()

    with torch.no_grad():

        _, enhanced_image, _ = DCE_net(
            data_lowlight
        )

    enhancement_time = (
        time.time() - start
    )

    return (
        enhanced_image.cpu(),
        enhancement_time
    )


# ============================================================
# EDGE DETECTION
# ============================================================

def apply_edge_style(image, style):

    if style == "None":
        return image

    gray = cv.cvtColor(
        image,
        cv.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Canny
    # --------------------------------------------------------

    if style == "Canny":

        edges = cv.Canny(
            gray,
            100,
            200
        )

    # --------------------------------------------------------
    # Sobel
    # --------------------------------------------------------

    elif style == "Sobel":

        sobel_x = cv.Sobel(
            gray,
            cv.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv.Sobel(
            gray,
            cv.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = cv.magnitude(
            np.float32(np.abs(sobel_x)),
            np.float32(np.abs(sobel_y))
        )

        edges = cv.convertScaleAbs(
            magnitude
        )

    # --------------------------------------------------------
    # Laplacian
    # --------------------------------------------------------

    elif style == "Laplacian":

        edges = cv.Laplacian(
            gray,
            cv.CV_8U
        )

    else:

        return image

    return edges


# ============================================================
# PSEUDO DEPTH
# ============================================================

def create_pseudo_depth(image):

    gray = cv.cvtColor(
        image,
        cv.COLOR_BGR2GRAY
    )

    depth = cv.normalize(
        gray,
        None,
        0,
        255,
        cv.NORM_MINMAX
    )

    depth = cv.applyColorMap(
        depth,
        cv.COLORMAP_INFERNO
    )

    return depth


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload Image")

file = st.file_uploader(
    "Choose a low-light image",
    type=[
        "jpeg",
        "jpg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if file is not None:

    # --------------------------------------------------------
    # Read original image
    # --------------------------------------------------------

    original_image = Image.open(
        file
    ).convert("RGB")


    # --------------------------------------------------------
    # Zero-DCE Enhancement
    # --------------------------------------------------------

    with st.spinner(
        "Enhancing low-light image..."
    ):

        enhanced_tensor, enhancement_time = (
            lowlight(
                original_image
            )
        )


    # --------------------------------------------------------
    # Clamp output
    # --------------------------------------------------------

    enhanced_tensor = torch.clamp(
        enhanced_tensor,
        0,
        1
    )


    # --------------------------------------------------------
    # Tensor -> NumPy
    # --------------------------------------------------------

    enhanced_np = (
        enhanced_tensor
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).astype(np.uint8)


    enhanced_image = Image.fromarray(
        enhanced_np
    )


    # Save enhanced image

    enhanced_image.save(
        "test.png"
    )


    # ========================================================
    # DENOISING OPTION
    # ========================================================

    processing_option = st.radio(
        "Image Processing",
        [
            "Denoising Image",
            "Without Denoising Image"
        ]
    )


    # --------------------------------------------------------
    # RGB -> BGR
    # --------------------------------------------------------

    enhanced_cv = cv.cvtColor(
        enhanced_np,
        cv.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    if processing_option == "Denoising Image":

        processed_cv = (
            cv.fastNlMeansDenoisingColored(
                enhanced_cv,
                None,
                denoise_strength,
                denoise_strength,
                7,
                21
            )
        )

        cv.imwrite(
            "test_uni.png",
            processed_cv
        )

        detection_input = "test_uni.png"

    else:

        processed_cv = enhanced_cv

        detection_input = "test.png"


    # ========================================================
    # APPLY YOLO PARAMETERS
    # ========================================================

    objectDetection.CONFIDENCE = (
        confidence
    )

    objectDetection.SCORE_THRESHOLD = (
        confidence
    )

    objectDetection.IOU_THRESHOLD = (
        iou_threshold
    )


    # ========================================================
    # YOLO OBJECT DETECTION
    # ========================================================

    with st.spinner(
        "Running YOLOv3 object detection..."
    ):

        detection_start = time.time()

        detected_image = (
            objectDetection.detection_recognition(
                detection_input
            )
        )

        detection_time = (
            time.time() - detection_start
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Results")


    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    with col1:

        st.image(
            original_image,
            width=300
        )

        st.caption(
            "Original Low-Light Image"
        )


    # --------------------------------------------------------
    # Enhanced
    # --------------------------------------------------------

    with col2:

        st.image(
            cv.cvtColor(
                enhanced_cv,
                cv.COLOR_BGR2RGB
            ),
            width=300
        )

        st.caption(
            "Zero-DCE Enhanced Image"
        )


    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    with col3:

        st.image(
            cv.cvtColor(
                detected_image,
                cv.COLOR_BGR2RGB
            ),
            width=300
        )

        st.caption(
            "YOLOv3 Object Detection"
        )


    # ========================================================
    # PROCESSING INFORMATION
    # ========================================================

    st.divider()

    st.header(
        "📈 Processing Information"
    )


    info1, info2, info3, info4 = (
        st.columns(4)
    )


    with info1:

        st.metric(
            "Confidence",
            "{:.2f}".format(
                confidence
            )
        )


    with info2:

        st.metric(
            "IoU Threshold",
            "{:.2f}".format(
                iou_threshold
            )
        )


    with info3:

        st.metric(
            "Enhancement Time",
            "{:.2f}s".format(
                enhancement_time
            )
        )


    with info4:

        st.metric(
            "Detection Time",
            "{:.2f}s".format(
                detection_time
            )
        )


    # ========================================================
    # EDGE VISUALIZATION
    # ========================================================

    if edge_style != "None":

        st.divider()

        st.header(
            "🔲 Edge Visualization"
        )


        edge_result = apply_edge_style(
            processed_cv,
            edge_style
        )


        if len(edge_result.shape) == 2:

            st.image(
                edge_result,
                width=700,
                caption=(
                    edge_style +
                    " Edge Detection"
                )
            )

        else:

            st.image(
                cv.cvtColor(
                    edge_result,
                    cv.COLOR_BGR2RGB
                ),
                width=700,
                caption=(
                    edge_style +
                    " Edge Detection"
                )
            )


    # ========================================================
    # DEPTH VISUALIZATION
    # ========================================================

    if (
        depth_estimator ==
        "Pseudo Depth (Brightness)"
    ):

        st.divider()

        st.header(
            "🗺️ Depth Visualization"
        )


        pseudo_depth = (
            create_pseudo_depth(
                processed_cv
            )
        )


        st.image(
            cv.cvtColor(
                pseudo_depth,
                cv.COLOR_BGR2RGB
            ),
            width=700,
            caption=(
                "Pseudo Depth Map "
                "(brightness-based; "
                "not a true depth estimator)"
            )
        )


    # ========================================================
    # CURRENT PARAMETERS
    # ========================================================

    with st.expander(
        "🔧 View Current Parameters"
    ):

        st.write(
            "Device:",
            str(DEVICE)
        )

        st.write(
            "Enhancement Model:",
            "Zero-DCE"
        )

        st.write(
            "Object Detection Model:",
            "YOLOv3"
        )

        st.write(
            "Confidence Threshold:",
            confidence
        )

        st.write(
            "IoU / NMS Threshold:",
            iou_threshold
        )

        st.write(
            "Denoising:",
            processing_option
        )

        st.write(
            "Denoising Strength:",
            denoise_strength
        )

        st.write(
            "Edge Style:",
            edge_style
        )

        st.write(
            "Depth Estimator:",
            depth_estimator
        )
