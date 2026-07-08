import streamlit as st
from PIL import Image

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Animal Image Classification",
    page_icon="🐾",
    layout="centered"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>

.stApp{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    color:#1E3A8A;
    font-size:40px;
    font-weight:bold;
}

.sub{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result{
    text-align:center;
    font-size:30px;
    color:green;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------

st.markdown("<div class='title'>🐾 Animal Image Classification</div>", unsafe_allow_html=True)

st.markdown("<div class='sub'>Upload an animal image and classify it using Machine Learning</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------


# ---------------------------
# Upload Image
# ---------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    if st.button("🔍 Predict", use_container_width=True):

        # Replace this with your prediction code
        prediction = "Cat"

        st.success("Prediction Completed Successfully!")

        st.markdown(
            f"<div class='result'>🐱 {prediction}</div>",
            unsafe_allow_html=True
        )

        st.balloons()

# ---------------------------
# Footer
# ---------------------------

st.write("")
st.markdown("---")

st.markdown(
    "<div class='footer'>Developed using Streamlit & Scikit-Learn</div>",
    unsafe_allow_html=True
)