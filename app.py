# Main page on course navigation
import streamlit as st
import pandas as pd
import numpy as np
import helper_functions as nav
import requests
import seaborn as sns
import matplotlib.pyplot as plt

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
    st.markdown(r"""
## Linear Regression (Chapter 3)

Linear regression is a very simple approach to *supervised learning*, particularly used to predict a quantitative response.
Since many of other (more fancy) statistical learning approaches are derived from linear regression, it is very important
to build a strong understanding of this model.
                
Here are some problems that Linear Regression would solve:

1. **Presence of Relationship**: Is there evidence that the variables are related? Does a variation in x cause any effect
in y?
2. **Relationship Strength**: How much a change in x explain a variation on y? 
3. **Relevant Variables**: Which other variables explain (or not) the variations of y?
4. **Impact Size**: How large is the impact of x over y? Couple pennies? Thousands of dolars?
5. **Prediction Accuracy**: For any given level of x, how accurately can we predict y?
6. **Linearity**: Verifies if rather or not the observation forms a linear pattern
7. **Interaction**: If x variables interact with themselves, would it result in a greater impact in y?
#### Simple Linear Regression (item 3.1)

It is very straight forward. Here we try to predict a quantitative response Y on the basis of a single preditor
X. It assumes that there is approximately a linear relationship between X and Y. Here is the equation:
$$
                Y \approx \beta_0 + \beta_1 X + \epsilon
$$
                
In this equation, $\beta_0$ represents the *Vertical Intercept*, and $\beta_1$ represents the *Slope*,
and they are called **Coefficients** or **Parameters**.

In practice, the Coefficients are unknown, and therefore we need to estimate them.


""")

    col1, col2, col3 = st.columns(3)
    # Data
    df = pd.read_csv('data/Advertising.csv')

    # Display in Streamlit
    with col1:
        sns.set_theme(style="darkgrid")
        width = 16
        height = 9
        fig = plt.figure(figsize=(height, width))
        ax = fig.add_subplot(1,1,1)
        sns.scatterplot(data=df, x='TV', y='sales', color='green')
        ax.set_ylim(0,30)
        ax.set_xlim(0,300)
        ax.set_xlabel('TV')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig.figure)
    with col2:
        sns.set_theme(style="darkgrid")
        width = 16
        height = 9
        fig = plt.figure(figsize=(height, width))
        ax = fig.add_subplot(1,1,1)
        sns.scatterplot(data=df, x='radio', y='sales', color='green')
        ax.set_ylim(0,30)
        ax.set_xlim(0,50)
        ax.set_xlabel('Radio')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig.figure)
    with col3:
        sns.set_theme(style="darkgrid")
        width = 16
        height = 9
        fig = plt.figure(figsize=(height, width))
        ax = fig.add_subplot(1,1,1)
        sns.scatterplot(data=df, x='newspaper', y='sales', color='green')
        ax.set_ylim(0,30)
        ax.set_xlim(0,100)
        ax.set_xlabel('Newspaper')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig.figure)

def show_cheatsheets():
    st.header('Cheat Sheet')

# # MAIN
if __name__ == "__main__":
    main()