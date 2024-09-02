import streamlit as st
from PIL import Image
from prediction import pred_class
import torch

# Set title 
st.title('Plant Disease Test')

# Set Header 
st.header('Please upload a picture')

# Load Model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

# Optional: Convert model to float16 if supported
# Uncomment the following line if you are sure that float16 is supported and needed
# model = model.to(dtype=torch.float16)

model = model.to(device)  # Ensure the model is on the correct device
model.eval()  # Set the model to evaluation mode

# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['fungal_bacterical', 'healthy', 'nutreint']

    if st.button('Predict'):
        try:
            # Prediction class
            predicted_class, prob = pred_class(model, image, class_name, device=device)
            
            st.write("## Prediction Result")
            st.write(f"**Class:** {predicted_class}")
            st.write(f"**Probability:** {prob * 100:.2f}%")
        
        except Exception as e:
            st.error(f"Error occurred during prediction: {str(e)}")
