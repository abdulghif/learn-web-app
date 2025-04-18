import streamlit as st

st.title('Hello Streamlit')

name = st.text_input("Enter your name", "Type Here...")
age = st.number_input("Enter your age", 0, 100, 25)

if st.button("Submit"):
    st.write(f"Hello {name}, you are {age} years old.")
