import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="Ornitho AI", layout="centered")

# Load your model (choose either the .h5 file or the Keras model file)
model = load_model('BirdModelFinal.h5')  # Replace with your model path

# Class labels for the bird species
class_labels = ['Crane', 'Crow', 'Egret', 'Kingfisher', 'Myna', 'Peacock', 'Pitta', 'Rosefinch', 'Tailorbird', 'Wagtail']

# Sidebar configuration
def sidebar():
    # Add a logo to the sidebar
    logo = Image.open("logo.jpeg")  # Replace with your logo file path
    st.sidebar.image(logo, use_column_width=True)

    # Add a brief description of the project
    st.sidebar.title("About our tool :)")
    st.sidebar.write("""
    Unleash the ornithologist within! Our AI-powered bird species classifier is your pocket-sized expert. 
    Simply upload an image, and watch as it swiftly identifies the feathered friend you've encountered. 
    From majestic cranes to vibrant kingfishers, our system has been trained to recognize a diverse array of avian wonders. 
    So, whether you're a seasoned birdwatcher or just starting your journey, let our tool assist you in exploring and appreciating 
    the beauty of nature's winged creatures.
    """)

# Navigation
def main():
    sidebar()  # Show the sidebar
    # Homepage
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        show_homepage()
    elif st.session_state.page == "classify":
        show_classification_page()

# Function to display the homepage
def show_homepage():
    st.title("Bird Species Classifier")

    st.write("""
    **Ornitho AI**
    This tool is an AI-powered image classification system that can predict the species of a bird from a given image.
    The model is trained to classify the following bird species:
    - Crane
    - Crow
    - Egret
    - Kingfisher
    - Myna
    - Peacock
    - Pitta
    - Rosefinch
    - Tailorbird
    - Wagtail

    Click on the "Classify" button below to start classifying bird species.
    """)

    if st.button("Classify"):
        st.session_state.page = "classify"
def display_prediction_with_facts(predicted_class):
    # Dictionary to store the facts for each bird species
    bird_facts = {
        "Crane": [
            "Migration Masters: Cranes are known for their long-distance migrations, often traveling thousands of miles each year.",
            "Dancing Divas: Many crane species perform elaborate courtship dances, which can include bowing, leaping, and spreading their wings."
        ],
        "Crow": [
            "Intelligent Aviators: Crows are considered highly intelligent birds, capable of complex problem-solving and tool use.",
            "Memory Champs: Crows have excellent memories and can recognize individual faces, even years later."
        ],
        "Egret": [
            "Fishery Friends: Egrets often follow cattle or buffalo through shallow water, preying on fish that are disturbed by the animals.",
            "Plumage Perfection: Egrets are known for their elegant white plumage, which they carefully preen and maintain."
        ],
        "Kingfisher": [
            "Hunting Heroes: Kingfishers are skilled hunters, diving into water to catch fish with their sharp beaks.",
            "Colorful Characters: Kingfishers come in a variety of vibrant colors, including blues, greens, and oranges."
        ],
        "Myna": [
            "Mimicry Masters: Myna birds are known for their ability to mimic human speech and other sounds.",
            "Social Creatures: Mynas are highly social birds and often live in large flocks."
        ],
        "Peacock": [
            "Eye-Catching Display: Male peacocks are famous for their elaborate tail feathers, which they spread out in a stunning display to attract females.",
            "Ancient Symbolism: Peacocks have been revered in many cultures throughout history and are often associated with royalty and immortality."
        ],
        "Pitta": [
            "Rainbow Birds: Pittas are known for their vibrant and colorful plumage, which often includes shades of blue, green, yellow, and red.",
            "Ground Dwellers: Most pitta species are ground-dwelling birds, foraging for insects and worms."
        ],
        "Rosefinch": [
            "Tiny Treasures: Rosefinches are small, colorful birds that are often found in mountainous regions.",
            "Social Birds: Rosefinches are often seen in pairs or small flocks, foraging for seeds and insects."
        ],
        "Tailorbird": [
            "Sewing Sensation: Tailorbirds get their name from their ability to stitch leaves together to create nests that resemble tiny purses.",
            "Insect Eaters: Tailorbirds are primarily insectivores, catching flies, moths, and other insects in flight."
        ],
        "Wagtail": [
            "Constant Motion: Wagtails are known for their distinctive wagging tails, which they constantly move up and down as they walk or run.",
            "Water-Loving Birds: Many wagtail species are found near water, where they forage for insects and small aquatic creatures."
        ]
    }

    # Display the predicted class
    st.write(f"You've spotted a **{predicted_class}**. Amazing !")

    # Display the "Did you know?" section
    st.write(f" Here are some 'Did you know?' facts about **{predicted_class}** for the Ornithologist in you to discover !")
    for fact in bird_facts[predicted_class]:
        st.write(f"- {fact}")
def show_classification_page():
    st.title("Species Classification")

    uploaded_file = st.file_uploader("Choose an image to classify!", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image successfully', use_column_width=True)
        st.write("")
        st.write("Let's wait for the model to classify...")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]

        # Display the predicted class and facts
        display_prediction_with_facts(predicted_class)
# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))  # Adjust size based on your model's input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Rescale the image if your model expects normalized input
    return image

if __name__ == "__main__":
    main()
