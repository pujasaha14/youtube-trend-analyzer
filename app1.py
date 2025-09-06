
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import chardet

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
def load_csv(file):
    rawdata=file.read()
    result=chardet.detect(rawdata)
    encoding=result['encoding']
    return pd.read_csv(pd.io.common.bytesIO(rawdata),encoding=encoding)
st.set_page_config(page_title="Youtube Trend Analyzer",layout="wide")
#Title
st.title("Youtube Trend Analyzer with ML")
st.write("Explore youtube trending videos datasetand predict video success using machine learning.")
#sidebar
st.sidebar.header("upload & settings")
upload_file=st.sidebar.file_uploader("upload your csv file",type=["csv"])
#load dataset
if upload_file:
    df=load_csv(upload_file)
else:
    df=pd.read_csv("youtube_trends_full.csv",encoding="utf-8")

#how dataset preview
st.subheader("Datadet preview")
st.dataframe(df.head())

#Data Visualization
#Top categories by views
if "category" in df.columns and "views" in df.columns:
    st.subheader("Top catgories by total views")
    category_views=df.groupby("category")["views"].sum().sort_values(ascending=False)
    fig, ax=plt.subplots(figsize=(10,5))
    sns.barplot(x=category_views.index, y=category_views.values,ax=ax)
    plt.xticks(rotation=45,ha="right")
    st.pyplot(fig)

#Likes vs comments scatter
if "likes" in df.columns and "comments" in df.columns:
    st.subheader("engagement: Likes vs Comments")
    fig, ax=plt.subplots(figsize=(7,5))
    sns.scatterplot(data=df,x="likes", y="comments",alpha=0.5)
    ax.set_title("Likes vs Comments")
    st.pyplot(fig)

#category Distribution pie
if "category" in df.columns:
    st.subheader('Category Distribution')
    category_counts=df["category"].value_counts()
    fig,ax=plt.subplots()
    ax.pie(category_counts.values, labels=category_counts.index,autopct="%1.1f%%",startangle=90)
    ax.set_title("video category distribution")
    st.pyplot(fig)

#Machine Learning part
st.header("Machine Learning Prediction")
if {"views","likes","comments"}.issubset(df.columns):
    df["popular"]=(df["views"]>df["views"].quantile(0.75)).astype(int)

    features=["likes","comments",]
    X=df[features]
    y=df["popular"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    model=RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    acc=accuracy_score(y_test,y_pred)


    st.subheader("Model performance")
    st.write(f"Accuracy: **{acc:.2f}**")
    st.text("classification report:")
    st.text(classification_report(y_test,y_pred))

    #user input for prediction
    st.subheader("predict video popularity")
    likes_input=st.number_input("Number of likes",min_value=0,value=1000)
    comments_input=st.number_input("Number of comments",min_value=0,value=200)

    if st.button("predict"):
        pred=model.predict([[likes_input,comments_input]])[0]
        if pred == 1:
            st.success("This video is likel to be popular!")
        else:
            st.warning("This video may not perform well")
else:
    st.warning("Dataset must have columns: views,likes,comment_count,category")

