import streamlit as st

def main():
    st.title("App 2")
    if st.session_state.get("clicked_button"):
        st.write("You clicked the button in App 1!")

if __name__ == "__main__":
    main()