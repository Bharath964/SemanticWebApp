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

# ------------------------
# Settings
# ------------------------
MODEL_PATH = "models/best_custom_unet_model.h5"
PATCH_SIZE = 256
N_CLASSES = 6
CONVERSION_FACTOR = 0.25

LABEL_MAPPING = {
    0: "Building",
    1: "Land",
    2: "Road",
    3: "Vegetation",
    4: "Water",
    5: "Unlabeled"
}

COLORS = {
    0: (60, 16, 152),
    1: (132, 41, 246),
    2: (110, 193, 228),
    3: (254, 221, 58),
    4: (226, 169, 41),
    5: (155, 155, 155)
}

scaler = MinMaxScaler()

def label_to_rgb(predicted_image):
    h, w = predicted_image.shape
    segmented_img = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, (r, g, b) in COLORS.items():
        segmented_img[predicted_image == lbl] = (r, g, b)
    return segmented_img

def generate_binary_mask(segmentation, target_label):
    return (segmentation == target_label).astype(np.uint8) * 255

def smooth_prediction(img, model):
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

    @st.cache_resource
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

    # Run prediction button
    if st.button("Run Prediction (Smooth Tiling)"):
        with st.spinner("Running smooth tiling prediction..."):
            smooth_pred = smooth_prediction(img_rgb, model)
            smooth_rgb = label_to_rgb(smooth_pred)
            st.session_state['smooth_pred'] = smooth_pred
            st.session_state['smooth_rgb'] = smooth_rgb
            st.session_state['prediction_done'] = True

    # Show Segmentation Output if prediction done
    if st.session_state.get('prediction_done', False):
        st.subheader("Segmentation Output")
        st.image(st.session_state['smooth_rgb'], caption="Smooth Tiling Segmentation", width=400)

        # Binary mask generation
        st.subheader("Generate Binary Masks and Compute Area")
        smooth_pred = st.session_state['smooth_pred']
        unique_labels = np.unique(smooth_pred)
        label_options = [f"{i}: {LABEL_MAPPING[i]}" for i in unique_labels]
        selection = st.multiselect("Select Classes", options=label_options)

        selected_ids = [int(s.split(":")[0]) for s in selection if s.strip()]

        if st.button("Generate Binary Masks"):
            combined_mask = np.zeros_like(smooth_pred, dtype=np.uint8)
            for label_id in selected_ids:
                combined_mask = np.logical_or(combined_mask, smooth_pred == label_id)
            combined_mask = (combined_mask.astype(np.uint8) * 255)
            st.image(combined_mask, caption="Combined Binary Mask", width=400)

            for label_id in selected_ids:
                class_name = LABEL_MAPPING.get(label_id, f"Label {label_id}")
                mask = generate_binary_mask(smooth_pred, label_id)
                labeled = sk_label(mask)
                regions = regionprops(labeled)
                st.markdown(f"**{class_name} (Label {label_id})**")
                total_area = sum(region.area for region in regions)
                real_area = total_area * CONVERSION_FACTOR
                st.write(f"Total Area: {total_area} pixels = {real_area:.2f} square units")
                if regions:
                    st.write(f"Number of Objects: {len(regions)}")

if __name__ == "__main__":
    main()
