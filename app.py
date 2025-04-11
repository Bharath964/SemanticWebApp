import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from simple_multi_unet_model import jacard_coef
from skimage.measure import label as sk_label, regionprops
from patchify import patchify, unpatchify

# ------------------------
# Settings
# ------------------------
MODEL_PATH = "models/best_custom_unet_model.h5"
PATCH_SIZE = 256
N_CLASSES = 6
CONVERSION_FACTOR = 0.25  # e.g., 1 pixel corresponds to 0.25 square meters

LABEL_MAPPING = {
    0: "Building",
    1: "Land",
    2: "Road",
    3: "Vegetation",
    4: "Water",
    5: "Unlabeled"
}

COLORS = {
    0: (60, 16, 152),    # Building: Dark blue/purple
    1: (132, 41, 246),   # Land: Purple
    2: (110, 193, 228),  # Road: Light blue
    3: (254, 221, 58),   # Vegetation: Yellow
    4: (226, 169, 41),   # Water: Orange
    5: (155, 155, 155)   # Unlabeled: Gray
}

scaler = MinMaxScaler()

def label_to_rgb(predicted_image):
    """Convert a 2D label image into an RGB image using the defined COLORS."""
    h, w = predicted_image.shape
    segmented_img = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, (r, g, b) in COLORS.items():
        segmented_img[predicted_image == lbl] = (r, g, b)
    return segmented_img

def generate_binary_mask(segmentation, target_label):
    """Generate a binary mask (0 or 255) for the given target label."""
    return (segmentation == target_label).astype(np.uint8) * 255

def postprocess_mask(mask, label_id):
    """Apply post-processing to mask based on label type."""
    if label_id == 0:  # Buildings - merge nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    elif label_id == 4:  # Water - remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def patchwise_prediction(img, model):
    """Perform patch‑wise (non‑overlapping) inference."""
    SIZE_X = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
    SIZE_Y = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE
    large_img = Image.fromarray(img).crop((0, 0, SIZE_X, SIZE_Y))
    large_img = np.array(large_img)
    patches_img = patchify(large_img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
    patched_prediction = []
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            patch = patches_img[i, j, 0, :, :, :]
            patch = scaler.fit_transform(patch.reshape(-1, 3)).reshape(patch.shape)
            patch = np.expand_dims(patch, 0)
            pred = model.predict(patch)
            pred = np.argmax(pred, axis=3)
            patched_prediction.append(pred[0])
    patched_prediction = np.array(patched_prediction)
    patched_prediction = np.reshape(patched_prediction,
                                    (patches_img.shape[0], patches_img.shape[1], PATCH_SIZE, PATCH_SIZE))
    return unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

def smooth_prediction(img, model):
    """Perform smooth tiling inference."""
    img_scaled = scaler.fit_transform(img.reshape(-1, 3)).reshape(img.shape)
    preds = predict_img_with_smooth_windowing(
        img_scaled,
        window_size=PATCH_SIZE,
        subdivisions=2,
        nb_classes=N_CLASSES,
        pred_func=lambda batch: model.predict(batch)
    )
    return np.argmax(preds, axis=2)

def main():
    st.set_page_config(layout="wide")
    st.title("Semantic Segmentation with Smooth Tiling and Area Computation")

    @st.cache(allow_output_mutation=True)
    def load_segmentation_model():
        return load_model(MODEL_PATH, compile=False)
    model = load_segmentation_model()

    uploaded_file = st.sidebar.file_uploader("Upload Aerial Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Input Image")
    st.image(img_rgb, caption="Original Image", width=400)

    if st.button("Run Prediction"):
        with st.spinner("Running patch-wise prediction..."):
            patch_pred = patchwise_prediction(img_rgb, model)
            patch_rgb = label_to_rgb(patch_pred)
        with st.spinner("Running smooth tiling prediction..."):
            smooth_pred = smooth_prediction(img_rgb, model)
            smooth_rgb = label_to_rgb(smooth_pred)
        st.session_state.update({
            'patch_pred': patch_pred,
            'patch_rgb': patch_rgb,
            'smooth_pred': smooth_pred,
            'smooth_rgb': smooth_rgb,
            'prediction_done': True
        })

    if st.session_state.get('prediction_done', False):
        st.subheader("Segmentation Outputs")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state['patch_rgb'], caption="Patch-wise Prediction", width=300)
        with col2:
            st.image(st.session_state['smooth_rgb'], caption="Smooth Tiling Prediction", width=300)

        st.subheader("Object Selection and Area Calculation")
        smooth_pred = st.session_state['smooth_pred']
        unique_labels = np.unique(smooth_pred)
        label_options = [f"{i}: {LABEL_MAPPING[i]}" for i in unique_labels]
        selected_labels = st.multiselect("Select Classes for Analysis", options=label_options)
        selected_ids = [int(s.split(":")[0]) for s in selected_labels if s.strip()]

        if st.button("Process Selected Classes"):
            processing_results = {}
            for label_id in selected_ids:
                mask = generate_binary_mask(smooth_pred, label_id)
                mask = postprocess_mask(mask, label_id)
                labeled = sk_label(mask)
                regions = regionprops(labeled)
                processing_results[label_id] = {
                    'mask': mask,
                    'regions': regions
                }
            st.session_state['processing_results'] = processing_results

        if 'processing_results' in st.session_state:
            processing_results = st.session_state['processing_results']
            for label_id in selected_ids:
                if label_id not in processing_results:
                    continue
                class_info = processing_results[label_id]
                regions = class_info['regions']
                class_name = LABEL_MAPPING.get(label_id, f"Label {label_id}")
                
                st.markdown(f"**{class_name} Analysis**")
                if not regions:
                    st.write(f"No objects detected for {class_name}")
                    continue
                
                region_options = {
                    f"Object {i+1} (Area: {reg.area * CONVERSION_FACTOR:.2f} units²)": reg
                    for i, reg in enumerate(regions)
                }
                selected_obj = st.selectbox(
                    f"Select {class_name} Object",
                    options=list(region_options.keys()),
                    key=f"obj_select_{label_id}"
                )
                chosen_region = region_options[selected_obj]
                
                # Display area information
                st.write(f"**Selected Area:** {chosen_region.area * CONVERSION_FACTOR:.2f} square units")
                
                # Visualize selected object
                mask = class_info['mask']
                highlighted = np.zeros_like(st.session_state['smooth_rgb'])
                highlighted[mask == 255] = st.session_state['smooth_rgb'][mask == 255]
                bbox = chosen_region.bbox
                cv2.rectangle(highlighted, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)
                st.image(highlighted, caption=f"Selected {class_name} Object", width=400)

if __name__ == "__main__":
    main()