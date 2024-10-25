# Follow these steps to run this application

# Step 1: WRITE BELOW PATH IN TERMINAL AND OPEN DIRECTORY
#   cd E:\Object_Detection_with_Yolo_CS619\3_Prototype_Phase

# Step 2: WRITE BELOW PATH IN TERMINAL AND RUN APPLICATION
#   streamlit run object_detection_app.py

# GET START DETECTION NOW! ;)

import streamlit as st
from streamlit_option_menu import option_menu
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import cv2

# Set page configuration first
st.set_page_config(page_title="Object Detection & Recognition",
                   layout='wide',
                   page_icon='./images/object.png')

# Menu options
selected = option_menu(
    menu_title=None,
    options=["Home", "Prediction"],
    icons=["house", "crosshair"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Display content based on selection
if selected == "Home":
    st.header("Final Project - Object Detection & Recognition")
    st.text('Group ID: S240200073 (BC200406553)')

    # Content
    st.markdown("""
    ### This App detects objects in real time from Images

    Our model detects three main categories in an image: 
    1. Human
    2. Animal
    3. Vehicle
    """)

elif selected == "Prediction":
    st.header('Get Object Detection from Image')
    st.write('Please Upload an Image to get detections')

    # Initialize the YOLO model once at the start
    with st.spinner('Loading the model...'):
        yolo = YOLO_Pred(onnx_model='models/best.onnx',
                         data_yaml='models/hamza_custom_data.yaml')

    def upload_image():
        image_file = st.file_uploader(label='Upload Image', type=['png', 'jpeg', 'jpg'])
        if image_file is not None:
            size_mb = image_file.size / (1024 ** 2)
            file_details = {
                "filename": image_file.name,
                "filetype": image_file.type,
                "filesize": "{:,.2f} MB".format(size_mb)
            }
            if file_details['filetype'] in ('image/png', 'image/jpeg'):
                st.success('VALID IMAGE file type (png or jpeg)')
                return {"file": image_file, "details": file_details}
            else:
                st.error('INVALID Image file type')
                return None

    def main():
        object = upload_image()

        if object:
            image_obj = Image.open(object['file'])

            # Convert to numpy array
            image_array = np.array(image_obj)

            # Check and convert from RGBA to RGB if necessary
            if image_array.shape[2] == 4:  # If the image has an alpha channel
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            # Get predictions
            with st.spinner("Getting objects from image, please wait..."):
                pred_img = yolo.predictions(image_array)

            # Convert the predicted image back to a PIL Image for display
            pred_img_obj = Image.fromarray(pred_img)

            # Create two columns
            col1, col2 = st.columns(2)

            # Display original image in the first column
            with col1:
                st.subheader("Original Image")
                st.image(image_obj, use_column_width=True)

            # Display predicted image in the second column
            with col2:
                st.subheader("Predicted Image")
                st.image(pred_img_obj, use_column_width=True)

    if __name__ == "__main__":
        main()