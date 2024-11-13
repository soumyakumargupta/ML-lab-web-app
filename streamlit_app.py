import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('iris_classification_notebook.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and App Description
st.title("Iris Data-set Prediction App")
st.markdown("""
Welcome to the **Iris Flower Species Prediction App**!  
This app predicts the species of an iris flower based on *sepal* and *petal* measurements.  
Simply adjust the sliders below to input measurements, and click *Predict* to see the species!
""")

# Sidebar with Guide and Feature Descriptions
st.sidebar.header("📋 Guide to Input Features")
st.sidebar.write("""
**Feature Descriptions:**  
- **Sepal Length**: The length of the sepal in centimeters.
- **Sepal Width**: The width of the sepal in centimeters.
- **Petal Length**: The length of the petal in centimeters.
- **Petal Width**: The width of the petal in centimeters.
  
**Instructions:**  
1. Use the sliders to match the flower's characteristics.
2. Click *Predict* to see the predicted species!
""")

# Input Fields for Flower Measurements
st.header(" Enter Flower Measurements")
st.write("Use the sliders below to input the sepal and petal dimensions of the flower:")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1, help="Select the sepal length of the flower")
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1, help="Select the sepal width of the flower")
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, step=0.1, help="Select the petal length of the flower")
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, step=0.1, help="Select the petal width of the flower")

# Species Characteristics for Better Prediction Understanding
st.header("Characteristics of Each Iris Species")
st.markdown("""
- **Iris-setosa**: Known for its smaller petals, with petal lengths usually below 2 cm.
- **Iris-versicolor**: Features intermediate petal sizes, with petal lengths typically between 3 to 5 cm.
- **Iris-virginica**: Recognizable for its large petals, with petal lengths often above 5 cm and widths over 1.5 cm.
""")

# Prediction button and Result Display
if st.button("Predict"):
    # Prepare input data for the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Perform prediction
    prediction = model.predict(input_data)
    
    # Map the prediction to species names
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    predicted_species = species[prediction[0]]
    
    # Display prediction results
    st.success(f"The predicted species is: **{predicted_species}** ")
    
    # Additional insights based on the prediction
    if predicted_species == "Iris-setosa":
        st.info("**Iris-setosa**: This species is generally characterized by smaller petal measurements.")
    elif predicted_species == "Iris-versicolor":
        st.info("**Iris-versicolor**: Known for having intermediate-sized petals, making it distinct from both Iris-setosa and Iris-virginica.")
    else:
        st.info("**Iris-virginica**: This species has larger petal dimensions, making it the largest among the three.")

    # Display measurements summary
    st.write("### Input Summary")
    st.write(f"**Sepal Length:** {sepal_length} cm")
    st.write(f"**Sepal Width:** {sepal_width} cm")
    st.write(f"**Petal Length:** {petal_length} cm")
    st.write(f"**Petal Width:** {petal_width} cm")

    # Provide visual guidance on species similarity
    st.write("### Additional Information")
    st.markdown("""
    - This prediction is based on a pre-trained model that has been tested on a classic Iris flower dataset.
    - Please note that predictions are based on general trends, and there may be overlap between species for certain measurements.
    """)
