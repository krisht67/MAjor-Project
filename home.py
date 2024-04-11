import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def main():
    # Set page title
    st.set_page_config(page_title="Interview Preparation Coach")

    # Add custom HTML and CSS for background image
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar navigation
    page = st.sidebar.radio("Go to", ["Home", "Interview", "Contact Us"])

    # Home page
    if page == "Home":
        st.title("Welcome to Interview Preparation Coach")
        if st.button("Start Interview"):
            switch_page("Interview")  # Switch to the Interview page

    # Interview page
    elif page == "Interview":
        st.title("Interview Page")
        st.write("You are on the Interview page.")

    # Contact Us page
    elif page == "Contact Us":
        st.title("Contact Us")
        st.write("You are on the Contact Us page.")


if __name__ == "__main__":
    main()
