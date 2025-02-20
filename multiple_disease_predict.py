import os
import pickle
import streamlit as st 
## Streamlit run command: streamlit run "E:\Multiple Disease Prediction System\multiple_disease_predict.py" 
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import h5py
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open(os.path.join('saved_model', 'diabetes_model.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join('saved_model', 'heart_disease_model.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join('saved_model', 'parkinsons_model.sav'), 'rb'))

# Breast Cancer Model Functions
class_mapping = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal',
}

@st.cache_resource
def load_breast_cancer_model():
    try:
        base_url = "https://raw.githubusercontent.com/Abdullah-Khan9653/Multiple_Disease_Prediction_System/main/Breast-Cancer-Image-Classification-with-DenseNet121/splitted_model/"
        model_parts = [f"{base_url}model.h5.part{i:02d}" for i in range(1, 35)]
        
        model_bytes = b''
        
        # Download and concatenate model parts
        for part_url in model_parts:
            response = requests.get(part_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download model part: {part_url}")
            model_bytes += response.content
        
        # Create a temporary file-like object
        model_buffer = BytesIO(model_bytes)
        
        custom_objects = {
            'Custom>Adam': tf.keras.optimizers.Adam,
            'Adam': tf.keras.optimizers.Adam
        }
        
        # Load the model using the buffer
        with h5py.File(model_buffer, 'r') as hf:
            model = tf.keras.models.load_model(
                hf,
                custom_objects=custom_objects,
                compile=False
            )
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_breast_cancer(image, model):
    if model is None:
        return "Error: Model not loaded"
    
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Ensure the image has 3 channels (RGB)
        if len(img_array.shape) == 2:  # If grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4:  # If RGBA
            img_array = img_array[..., :3]
        
        # Resize and preprocess
        img_array = tf.image.resize(img_array, (256, 256))
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_mapping[np.argmax(predictions[0])]
        return predicted_class
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'MediScan AI',  
        ['Diabetes Scan',
         'Cardiac Analysis',
         'Movement Disorder Check',
         'Breast Cancer Detection'],  # Added new option
        menu_icon='robot',  
        icons=['activity', 'heart-pulse', 'person-lines-fill', 'shield-fill-check'],  
        default_index=0,
        styles={
            "container": {"padding": "4px", "background-color": "#1a1a1a"},
            "icon": {"color": "#6ee7b7", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "font-weight": "normal",
                "color": "#FAFAFA",
                "border-radius": "5px",
                "--hover-color": "#2d2d2d"
            },
            "nav-link-selected": {
                "background-color": "#2d2d2d",
                "color": "#6ee7b7",
                "font-weight": "bold"
            },
            "menu-title": {
                "color": "#6ee7b7",
                "font-size": "20px",
                "font-weight": "bold"
            }
        }
    )


# Diabetes Prediction Page
if selected == 'Diabetes Scan':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

  # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Cardiac Analysis':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        

        

    st.success(heart_diagnosis)


# Parkinson's Prediction Page
if selected == "Movement Disorder Check":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# Breast Cancer Detection Page
if selected == "Breast Cancer Detection":
    st.title('Breast Cancer Detection using ML')
    st.write("Upload a breast ultrasound image for classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_container_width=True)

            with st.spinner('Loading model...'):
                model = load_breast_cancer_model()

            if model is not None:
                with st.spinner('Making prediction...'):
                    predicted_class = predict_breast_cancer(image, model)
                    st.success(f"Prediction: {predicted_class}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")