# Main page on course navigation
import streamlit as st
import helper_functions as nav


# # INITIALIZE SESSIONS & DATASETS

# # FUNCTIONS
def main():
    st.title("📖 CS109x Walkthrough")

    st.markdown(
        """
"""
    )
    menu = list(nav.classes_dict.keys())
    
    # Sidebar navigation
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice in nav.classes_dict:
        nav.classes_dict[choice]()



# # MAIN
if __name__ == "__main__":
    main()