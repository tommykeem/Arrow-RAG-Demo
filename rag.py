# Import necessary libraries
import os 
import base64
import streamlit as st
from streamlit_theme import st_theme 

# Import RAG engine functions for different deployment options and data sources
from rag_engine import (
    local_get_answer_folder_pdf,
    local_get_answer_upload_pdf, 
    local_get_answer_url,
    nim_get_answer_folder_pdf,
    nim_get_answer_upload_pdf,
    nim_get_answer_url,
    get_model
)

# Get current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Configure Streamlit page settings
st.set_page_config(
    page_title="Arrow AI Assistant",
    page_icon = "./static/img/favicon.png",
    layout="centered"
)

# Background image handling functions
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    """Convert binary file to base64 string"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_photo_as_bg(png_file):
    """Set background image for the Streamlit app"""
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# Set background image
set_photo_as_bg('./static/img/racecars.jpg')

# Load custom CSS stylesheet
with open('./static/stylesheet/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    # css = f.read()


# Define logo paths and handle theme-based styling
lightmode_logo = "./static/img/arrow_worm_black.png"
darkmode_logo = "./static/img/arrow_worm_white.png"
darkmode_sidebar_logo = "./static/img/favicon.png"
theme = st_theme()

# Apply theme-specific styling (light/dark mode)
if theme is not None and theme["base"] == "light":      # Light mode styling
    st.logo(lightmode_logo, icon_image=lightmode_logo)
    st.markdown("""
    <style>
        [data-testid='stFileUploader'] label{color: white !important;} /* color of the file uploader label */
        [data-testid='stFileUploaderFileName'] {color: white !important;} /* color of the file name */
        .st-emotion-cache-1aehpvj.e1bju1570 {color: gray !important;} /* color of the file size */
        [data-testid="stFileUploaderDeleteBtn"] {color: white !important;} /* color of the delete button */
        /*  [data-testid='stWidgetLabel'] p {color: white !important;} color of the widget label */
        /*  .st-emotion-cache-uzeiqp p {color: white !important;} color of the widget label */
        .st-an.st-ap {background-color: lightgray !important;} /* color of the drop down menu on the side bar */
        [data-testid='stException'] {   /* color of the exception error box */
            background-color: white !important;
            border-radius: 10px !important;
        }
        [data-testid='stException'] p, ul, ol{  color: red !important;} /* color of the exception error text */
        [data-testid='stAlert']{   /* color of the alert box */
            background-color: white !important;
            border-radius: 10px !important;
        }
        [data-testid='stAlert'] p, ul, ol {  color: black !important;} /* color of the alert text */
       
        
    </style>
    """, unsafe_allow_html=True)
else:                                                   # Dark mode styling 
    st.logo(darkmode_logo, icon_image=darkmode_logo)
    st.markdown("""
    <style>
        .st-emotion-cache-1sno8jx p{color: white !important;} /* color of the sidebar text */
         [data-testid='stException'] {   /* color of the exception error box */
            background-color: black !important;
            border-radius: 10px !important;
        }
        [data-testid='stException'] p, ul, ol{  color: red !important;} /* color of the exception error text */
        [data-testid='stAlert']{   /* color of the alert box */
            background-color: black !important;
            border-radius: 10px !important;
        }
        [data-testid='stAlert'] p, ul, ol {  color: white !important;} /* color of the alert text */    
    </style>
    """, unsafe_allow_html=True)
    

# Display RAG diagram image
st.image('./static/img/rag_diagram.png', width= 700)


# Sidebar configuration
with st.sidebar:
    # Deployment options dropdown
    deployment = st.selectbox('Select Deployment Option',( 'DGX Local Deployment', 'NVIDIA NIM'))
    
    # Database type selection
    database = st.selectbox('Select RAG Database',('Local PDFs (.pdf)', 'Upload PDFs (.pdf)', 'Public Websites (URL)'))
    
    # Model selection
    model = st.selectbox('Select LLM Model',('Meta Llama 3.1', 'Google Gemma 2', 'Microsoft Phi 3.5'))
   
    # Temperature slider for model creativity
    # Define the range for the slider
    min_value = 0.0
    max_value = 2.0
    step = 0.1  # Optional: define the step size for the slider
    
    # Create the slider for float values between 0 and 2
    temperature = st.slider("Select LLM's Creativity Level", min_value, max_value, 0.4)
    temperature_text = str(temperature)
    
    # Custom styling for slider min and max labels
    st.markdown("""
    <style>
        [data-testid='stTickBarMin'] {
            display:none !important;
            padding: 0px !important;
            margin: 0px !important; /* hide slider min label */
        }
        [data-testid='stTickBarMax'] {
            display:none !important;
            padding: 0px !important;
            margin: 0px !important; /* hide slider max label */
        }
        [data-testid='stSlider'] { /* Adjust the slider's margin */
            margin-top: 20px !important;
        }
        .label-container {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .label {
            padding: 0px;
            margin: -20px 0 0 0;
        }
    </style>
    <div class='label-container'>
        <h6 class='label'>Structured</h6>
        <h6 class='label'>Creative</h6>
    </div>
    """, unsafe_allow_html=True)

    st.write('You selected:\n\n**Deployment Method:** %s\n\n**Database:** %s\n\n**LLM:** %s\n\n**LLM Temperature:** %s' % (deployment, database, model, temperature_text))

# Handle DGX Local Deployment
if deployment == "DGX Local Deployment": 
    if database == 'Local PDFs (.pdf)':             # Handle local PDF folder queries
        user_query = st.text_input(" ")
        if st.button("Ask") or user_query:

            model = get_model(model,deployment)
            answer = local_get_answer_folder_pdf(model, temperature, user_query)

            st.success(answer)    
    elif database == 'Upload PDFs (.pdf)':          # Handle PDF file upload and queries
       
        uploaded_file = st.file_uploader(label="Upload your files", type=["pdf"])

        user_query = st.text_input(" ")

        if st.button("Ask") or user_query:
        
            bytes_data = uploaded_file.read()
            file_name = uploaded_file.name

            # save the upoloaded file to the working directory
            file_path = os.path.join(working_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(bytes_data)

            model = get_model(model,deployment)
            answer = local_get_answer_upload_pdf(model, temperature, file_name, user_query)

            st.success(answer)
    elif database == 'Public Websites (URL)':       # Handle URL-based queries
    
     
        user_query = st.text_input(" Enter your question ")

        if st.button("Ask") or user_query:
    
            model = get_model(model,deployment)
            answer = local_get_answer_url(model, temperature, user_query)

            st.success(answer)

# Handle NVIDIA NIM Deployment
if deployment == "NVIDIA NIM":
    if database == 'Local PDFs (.pdf)':             # Handle local PDF folder queries with NIM
        
        user_query = st.text_input(" ")
        
        if st.button("Ask") or user_query:
        
            model = get_model(model,deployment)
            answer = nim_get_answer_folder_pdf(model, temperature, user_query)

            st.success(answer)         
    elif database == 'Upload PDFs (.pdf)':          # Handle PDF file upload and queries with NIM
        uploaded_file = st.file_uploader(label="Upload your files", type=["pdf"])

        user_query = st.text_input(" ")

        if st.button("Ask") or user_query:
        
            bytes_data = uploaded_file.read()
            file_name = uploaded_file.name

            # save the upoloaded file to the working directory
            file_path = os.path.join(working_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            model = get_model(model,deployment)
            answer = nim_get_answer_upload_pdf(model,temperature, file_name, user_query)

            st.success(answer)          
    elif database == 'Public Websites (URL)':       # Handle URL-based queries with NIM
    
        # url_database = st.text_input(" Enter the URL ")
        user_query = st.text_input(" Enter your question ")

        if st.button("Ask") or user_query:
            model = get_model(model,deployment)
            answer = nim_get_answer_url(model,temperature,user_query)

            st.success(answer)
