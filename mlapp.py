import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import io

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

st.title("Rice - Cammeo or Osmancik?")
st.write("Ingrid Hansen - r0879034")

st.sidebar.header("Select a Machine Learning Model")
selected_model = st.sidebar.selectbox("Choose a Model", ["Random Forest", "K-Nearest Neighbors", "Support Vector Machine"])

st.write("These machine learning models are trained to be able to predict which species of rice an instance is depending on their features.\nThe models available in this application are Random Forest, K-nearest Neighbor and Support Vector Machines.")
#Read the data from the file
data = pd.read_csv('rice+cammeo+and+osmancik/rice.csv', delimiter=',',
                    names=['Area','Perimeter','Major Axis Length',
                           'Minor Axis Length','Eccentricity','Convex_Area','Extent','Species'])

feature_cols = ['Area', 'Perimeter', 'Major Axis Length', 'Minor Axis Length', 'Eccentricity', 'Convex_Area', 'Extent']
X = data[feature_cols]
y = data['Species']

defaul_value = 52
random_state = st.number_input("Enter a new number to change the model datasets:",min_value=1,max_value=100, value=52)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)

# Show data summary
st.write("## Data Summary")
st.write("Shape of the dataframe:", data.shape)
with st.expander("Statistical Information about the dataframe"):
    st.write(data.describe())
with st.expander("Information about the dataframe:"):
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
with st.expander("First 5 instances in the dataframe"):
    st.write(data.head())
with st.expander("Last 5 instances in the dataframe"):
    st.write(data.tail())
# Data Visualization
st.write("## Data Visualization")
st.bar_chart(data['Species'].value_counts(), use_container_width=True)

# Model Training and Evaluation
if selected_model == "Random Forest":
    st.write("## Random Forest Model")

    # Define the number of trees for random forest
    number_of_trees = k = st.slider("Select number of trees:",min_value=50, max_value=400, value=100)
    rfc = RandomForestClassifier(n_estimators=number_of_trees)
    rfc.fit(X_train, y_train)
    f_pred = rfc.predict(X_test)

    st.write("### Accuracy:", metrics.accuracy_score(y_test, f_pred))

    # Create confusion matrix

    target = np.array(y_test)
    forest = np.array(f_pred)
    cm = confusion_matrix(target, forest)
    st.write("### Confusion Matrix:")
    st.write(cm)

    # Plot the confusion matrix
    st.pyplot(sns.heatmap(cm, annot=True, fmt="d", cmap="Blues").get_figure())

elif selected_model == "K-Nearest Neighbors":
    st.write("## K-Nearest Neighbors Model")

    k = st.slider("Select a neighbour value:",min_value=1, max_value=100, value=3)
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X_train, y_train)
    kn_pred = kn.predict(X_test)
    kn_accuracy = accuracy_score(y_test, kn_pred)

    st.write("### Accuracy:", kn_accuracy)

    # Create confusion matrix
    target = np.array(y_test)
    kneighbour = np.array(kn_pred)


    kn_cm = confusion_matrix(target, kneighbour)
    st.write("### Confusion Matrix:")
    st.write(kn_cm)

    # Plot the confusion matrix
    st.pyplot(sns.heatmap(kn_cm, annot=True, fmt="d", cmap="Blues").get_figure())
    

elif selected_model == "Support Vector Machine":
    st.write("## Support Vector Machine Model")

    kernel = 'linear'  # You can change the kernel here
    svm_classifier = SVC(kernel=kernel, decision_function_shape='ovr')
    svm_classifier.fit(X_train, y_train)
    svm_pred = svm_classifier.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)

    st.write("### Accuracy:", svm_acc)

    #Create the confusion matrix
    target = np.array(y_test)
    svma = np.array(svm_pred)

    svm_cm = confusion_matrix(target, svma)
    st.write("### Confusion Matrix:")
    st.write(svm_cm)

    # Plot the confusion matrix
    st.pyplot(sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues").get_figure())