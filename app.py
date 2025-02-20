# Main page on course navigation
import streamlit as st
import pandas as pd
import numpy as np
import helper_functions as nav
import requests

from datetime import datetime
from ISLP import *


# # INITIALIZE SESSIONS & DATASETS


# # FUNCTIONS
def main():
    st.title("📖 CS109x Walkthrough")

    st.markdown(
        """
"""
    )
    menu = [
        'Introduction',
        'Class 1: Simple Regression',
        'Cheat Sheets'
    ]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == menu[0]:
        show_introduction()
    elif choice == menu[1]:
        show_class1()
    elif choice == menu[-1]:
        show_cheatsheets()

def show_introduction():
    st.markdown("""
**HarvardX Course:** [edX](https://www.edx.org/learn/data-science/harvard-university-introduction-to-data-science-with-python)

**ISLP Website:** [statlearning.com](http://www.statlearning.com/)
""")
    st.header('Welcome to this Project!')
    st.text('This page is e brief introduction and explanation of what we will work along the course')
    st.markdown("""
### The Course
Introduction to Data Science with Python, presented by Professor Pavlos Protopapas,
is an *intermediate* course, so it requieres
basic knowledge on statistics, calculus, linear algebra and, of course, Python3 syntax.
Since this is a Walkthrough, I will skip the most complicated concepts, but in the final 
page, I will add a cheadsheet with the most important models and equations, using LaTeX.

The Manual Book for this course is *An Introduction to Statistical Learning, with Applications in Python* (ISLP),
it is also pointed out the importance of it's precursos: *Elemets of Statistical Learning* (ESL), but since this
book has more complicated concepts and derivations, we will be using only ISLP.

For more detailed explanations of each modules, you can check the **notepad** subfolder from branch main, 
there is there I add all my notes during the pre-reading.

### Introduction (Preface, Chapter 1)
Statistical Learning refer to various set of tools for **understanding data**.
These tools can be classified in 'supervised' and 'unsupervised':
- Supervised: Predicting or estimating an *output* based on one or more *inputs*.
(e.g. Business, Medicine, Astrophysics, Public Policy, etc)
- Unsupervised: involves inputs *without* known outputs, allowing us to discover 
relationships and structures within the data (e.g., DNA genome tracking).

To better illustrate some of the concepts, we will be using some of the datasets from the **ISLP** python package. 
The Manual also mentions some of the history of Statistical Leaning, but I will leave it to the notebook .txt files.

### Understanding the problem: how to apply the solution?
First of all, in the matter of Statistics, there are four types os analytics:
- Descriptive: Provides information about what happened in the past and give a clear view of the current state.
- Diagnostic: Helps understand why something happened in the Past, along with exploring relationships and 
                correlations among the variables.
- Predictive: Seek trends and tendencies, predicting what is most likely to happen in the future.
- Prescriptive: Recommends actions to effectively affect those predicted outcomes.

Then, we need to identify the type of data we are dealing with so we can apply the models. There are three main types of problems:
- Regression: Involves a *continuous* or *quantitative* output. (e.g., Wage, Weigth, Population Growth)
- Classification: Involves a non-numerical values, for *categorical* or *qualitative* outputs. (e.g., Iris, Stock Market Movements [bool])
- Clusterization: When we only have input variables, but no known output, just like the unsupervised problem. (e.g. Genome Tracking, Mall Client Types)

Finally, we need to decide which models will better suit our need in the given problem, 
and use *Cross-validation* and *bootstrap* to validate the fitting of the chosen models.

I will make sure to provide a clear explanation of each, in a structured and slow paced manner.
                
Now, let's begin with Simple Linear Regressions! 🚀
""")

def show_class1():
    st.markdown("""
## Linear Regression (Chapter 3)
""")


def show_cheatsheets():
    st.header('Cheat Sheet')

# # MAIN
if __name__ == "__main__":
    main()