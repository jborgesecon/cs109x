# Main page on course navigation
import streamlit as st
import pandas as pd
import numpy as np
import helper_functions as nav
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


from datetime import datetime
from ISLP import *
from ISLP.models import (ModelSpec as MS, summarize, poly)

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

In practice, the Coefficients are unknown, and therefore we need to estimate them (finding the intercept
$\beta_0$ and the slope $\beta_1$).

Our goal is to calculate the least possible distance among all the observations when tracing a line,
there are a couple ways of doing so, but here we are studying the *least squares*.

#### Least Squares

Basically, imagine all the observations, now we trace a straight line anywhere in this graph. Now, we
measure the distance of each observation to the line and write them down. After the last observation, 
we sum them up, so we see the "net distance" of all the observations regarding our line. This **could**
help us find the best possible position for the line, seeking the least possible net distance, but this
approach is flawed: observations above the line would represent a negative distance, and this biases
our study. The solution found by our dear precursors was to square all the distances, so all the 
values would be positive, effectively solving our problem.

The "net distance" I mentioned is actually called *residual sum of squares*, but I won't derive it's 
formula here.

We can see some of those concepts graphically, and so I'm going to use the 'Advertising.csv' dataset,
provided by the authors. This dataset has the sales, and the investment on TV, radio and newspaper, 
below are the scatterplots of those variables, having 'sales' as the dependent (y):

""")
    

    col1, col2, col3 = st.columns(3)
    # Data
    df = pd.read_csv('data/Advertising.csv')

    # Display in Streamlit (since pratically all data is customized, I chose not to make a def)
    with col1:
        sns.set_theme(style="darkgrid")
        width = 16
        height = 9
        fig = plt.figure(figsize=(height, width))
        ax = fig.add_subplot(1,1,1)
        sns.scatterplot(data=df, x='TV', y='sales', color='blue')
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
        sns.scatterplot(data=df, x='radio', y='sales', color='blue')
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
        sns.scatterplot(data=df, x='newspaper', y='sales', color='blue')
        ax.set_ylim(0,30)
        ax.set_xlim(0,100)
        ax.set_xlabel('Newspaper')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig.figure)

    st.markdown("""
First, let's analyse the TV regressor. According to the formula provided earlier, we need to find
the intercept and the slope. The code snippet used to find those values are on the **response.ipynb**
and can be checked there.
                
Here is the result as a table, with not only the *coefficients*, but also the *standard error*, 
the *t-student* value, and the *P-value* (We'll learn more about them soon):
""")
    
    def coefficients(df, x, y):
        x1 = pd.DataFrame({
            'Intercept': np.ones(df.shape[0]),
            'Slope': df[x]
            })
        y1 = df[y]
        model = sm.OLS(y1, x1)
        result = model.fit()

        return summarize(result)

    st.dataframe(coefficients(df=df, x='TV', y='sales'))

    st.markdown(r"""
Now that we found the $\beta_0$ and $\beta_1$, we can swap them in the OLS equation, resulting in:

$sales \approx 7.0326 + 0.0475 * TV$
                
#### We get the following regression line:
""")

    sns.set_theme(style="darkgrid")
    width = 16
    height = 9
    fig = plt.figure() #figsize=(height, width))
    ax = fig.add_subplot(1,1,1)

    # Scatterplot
    sns.scatterplot(data=df, x='TV', y='sales', color='blue')

    # Regression line
    df['predicted_sales'] = 7.0326 + 0.0475 * df['TV']  # Calculate predictions
    ax.plot(df['TV'], df['predicted_sales'], color='red', linewidth=2, label='Regression Line')

    # Residuals
    for i in range(len(df)):
        ax.plot([df['TV'][i], df['TV'][i]], [df['sales'][i], df['predicted_sales'][i]], color='blue', linewidth=0.4)

    # Set axis limits and labels
    ax.set_ylim(0, 30)
    ax.set_xlim(0, 300)
    ax.set_xlabel('TV')
    ax.set_ylabel('Sales')
    ax.legend()
    st.pyplot(fig.figure)

    st.markdown(r"""
The book then further explains the presenct of the $\epsilon$ term of error. It is actually very
interesting the way they explain it, so I do reccomend checking it out, but basically what they
do is: they create two variables x and y, and apply OLS to them, finding the interecept and the
slope. This would be the **True** values of both $\beta_0$ and $\beta_1$, but then they add an 
error term,as a normal distribution with mean equals zero, and then they perform again the OLS, 
now with the errors, and we can see that, altought the model does cath the tendency correctly, it 
is slightly changes both the slope and the intercept. I real-world situations, we will rarely have
the "true", but we can apply those methods we learn here to minimize this impact and define the
accuracy of the Estimates.


""")



def show_cheatsheets():
    st.header('Cheat Sheet')

# # MAIN
if __name__ == "__main__":
    main()