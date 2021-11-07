from math import gamma
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms are edible or poisonous ? 🍄")
    st.sidebar.markdown("Are your mushrooms are edible or poisonous? 🍄")
    
    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv(r'C:\Users\APOORVA\Desktop\MACHINE LEARNING\PROJECTS\Build a Machine Learning Web App with Streamlit and Python\mushrooms.csv')
        le=LabelEncoder()
        for col in data.columns:
            data[col]=le.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y=df.type
        X=df.drop(['type'],axis=1)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
        return X_train,X_test,y_train,y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test,y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test,y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, X_test,y_test)
            st.pyplot()

    df = load_data()
    X_train,X_test,y_train,y_test=split(df)
    class_names=['edible','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier=st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)","Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C=st.sidebar.number_input("C (Regularization Parameter)", 0.01,10.0,step=0.01, key='C')
        kernel=st.sidebar.radio("Kernel",("linear","rbf","sigmpid","poly"),key='kernel')
        gamma=st.sidebar.radio("Gamma (Kernel Coefficients)", ("scale","auto"),key='gamma')
        degree=st.sidebar.number_input("Degree (Degree of polynomial function)",1,10,step=1,key='degree')  

        metrics=st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Support Vector Machine (SVM) results")
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_train,y_train)
            accuracy=model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ",precision_score(y_test,y_pred, labels=class_names).round(2))
            st.write("Recall : ", recall_score(y_test,y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C=st.sidebar.number_input("C (Regularization Parameter)", 0.01,10.0,step=0.01, key='C_LR')
        max_iter =st.sidebar.slider("Maximum number of iteration",100,500,key='max_iter')

        metrics=st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Logistic Regression results")
            model=LogisticRegression(C=C,max_iter=max_iter)
            model.fit(X_train,y_train)
            accuracy=model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ",precision_score(y_test,y_pred, labels=class_names).round(2))
            st.write("Recall : ", recall_score(y_test,y_pred, labels=class_names).round(2))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators=st.sidebar.number_input("The number of trees in the forest",100,500,step=10,key='n_estimator')
        max_depth=st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key='max_depth')
        bootstrap=st.sidebar.radio("Bootstrap samples when building trees",('True','False'),key='bootstrap')
        criterion=st.sidebar.radio("Criterion",('entropy','gini'),key='criterion')

        metrics=st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Random Forest results")
            model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion,bootstrap=bootstrap,n_jobs=-1)
            model.fit(X_train,y_train)
            accuracy=model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ",precision_score(y_test,y_pred, labels=class_names).round(2))
            st.write("Recall : ", recall_score(y_test,y_pred, labels=class_names).round(2))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_metrics(metrics)




    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

    






if __name__ == '__main__':
    main()


