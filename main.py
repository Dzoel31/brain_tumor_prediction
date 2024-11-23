import streamlit as st
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pandas as pd

model = load_model('brain_tumor_model.h5')

st.set_page_config(page_title="Tumor Prediction", page_icon=":brain:")

st.header(":brain: Brain Tumor Prediction :mag:")

st.subheader("Upload a brain image")

file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

dict_label = {
    1: 'meningioma',
    2: 'glioma',
    3: 'pituitary'
}

if file:
    img = image.load_img(file, target_size=(150,150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    st.image(img, caption='Uploaded Image')

    prediction = model.predict(x)

    index_class = np.argmax(prediction)

    st.write("Result: ")

    result_df = pd.DataFrame({
        "Class": [value for key, value in dict_label.items()],
        "Probability": prediction[0]
    })

    result_df = result_df.sort_values(by='Probability',ascending=False).reset_index(drop=True)

    st.write(f"The image is classified as: {dict_label[index_class+1]}")

    result_df['Probability'] = result_df['Probability'].apply(lambda x: f"{x:.2f}")

    st.table(result_df)

