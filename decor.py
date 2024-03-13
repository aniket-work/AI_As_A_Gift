import streamlit as st
from PIL import Image, ImageFilter
from OllamaModelLoader import OllamaModelLoader
from image_utils import convert_image_to_base64, display_base64_image

# App title and description
st.set_page_config(page_title="Home Decor AI Assistant", page_icon=":house:")

# Add a blurred background image
background_image = Image.open("background.png")
blurred_background = background_image.filter(ImageFilter.GaussianBlur(radius=3))
st.image(blurred_background, use_column_width=True)

# Add text on top of the blurred image
st.markdown(
    """
    <div style="position: relative;">
        <h1 style="color: white; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); text-align: center; margin-top: -150px;">Home Decor AI Assistant</h1>
        <p style="color: white; font-size: 20px; font-weight: bold; text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.5); text-align: center; margin-top: -20px;">Get personalized recommendations for your dream home decor.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# upload images
def upload_image():

    # Use Markdown syntax to set font size
    st.markdown("<span style='font-size: 20px;'>Submit a visual and let our AI assist with your home styling!</span>",
                unsafe_allow_html=True)

    # Use file_uploader as usual
    images = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    assert len(images) <= 4, (st.error("Please upload at most 4 images"), st.stop())

    if images:
        # convert images to base64
        images_b64 = []
        for image in images:
            image_b64 = convert_image_to_base64(image)
            images_b64.append(image_b64)

        # display images in multiple columns
        cols = st.columns(len(images_b64))
        for i, col in enumerate(cols):
            col.markdown(f"**Image {abs((i+1)-len(cols))+1}**")
            col.markdown(display_base64_image(images_b64[i]), unsafe_allow_html=True)
        st.markdown("---")
        return images_b64
    st.stop()


# init session state of the uploaded image
image_b64 = upload_image()


# ask question
q = st.chat_input("Ask a question about the image(s)")
if q:
    question = q
else:
    # if isinstance(image_b64, list):
    if len(image_b64) > 1:
        question = f"Describe the {len(image_b64)} images:"
    else:
        question = "Describe the image:"


# load Ollama model using OllamaModelLoader
config_file_path = 'ollama_config.json'  # Path to the JSON configuration file
ollama_loader = OllamaModelLoader(config_file_path)
mllm = ollama_loader.load_ollama_model()

# Run the language model
@st.cache_data(show_spinner=False)
def run_language_model(query, image_base64):
    """
    Runs the language model with the given query and image context.

    Args:
        query (str): The user's question or prompt.
        image_base64 (str): The base64-encoded image data.

    Returns:
        str: The language model's response.
    """
    llm_with_image_context = mllm.bind(images=image_base64)
    response = llm_with_image_context.invoke(query)
    return response


# Display the user's question
with st.container():
    st.markdown(
        f"""
        <div style="background-color: #f0f0f0; padding: 16px; border-radius: 8px;">
            <h3 style="margin-top: 0;">Your Question:</h3>
            <p style="font-size: 16px; font-weight: bold;">{question}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Define the custom spinner
spinner_html = """
    <style>
        @keyframes spinner-rotation {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        .spinner {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            border: 8px solid #f3f3f3;
            border-top: 8px solid #3498db;
            animation: spinner-rotation 1s linear infinite;
        }
    </style>
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
        <div class="spinner"></div>
    </div>
"""

# Run the language model and display the response
with st.container():
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
    response = run_language_model(question, image_b64)
    spinner_placeholder.empty()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="background-color: #e6f7ff; padding: 16px; border-radius: 8px;">
            <h3 style="margin-top: 0;">Response:</h3>
            <p style="font-size: 16px;">{response}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
