import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from ISLP import *
from ISLP.models import (ModelSpec as MS, summarize, poly)

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

    st.text('\n')

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

Those can be calculated as the *Residual Standard Error* (RSE) formula, that involves the 
residual sum of squares (RSS), the 
standard deviation ($\sigma^2$) and amostration size (*n*).

The standard error are used to find the *confidence interval*. A 95% confidence interval means that
there is a 95% change that the given range will contain the true value for the parameter. 
This is also called *Type I Error*, here is a table below regaring the errors:
""")

    # Define the data
    type_errors = {
        'Null Hypothesis': ['True', 'False'],
        'Accept': ['Correct', 'Type II Error'],
        'Reject': ['Type I Error', 'Correct']
    }

    # Create the DataFrame
    df = pd.DataFrame(type_errors).set_index('Null Hypothesis')

    # Define a function for styling
    def highlight_cells(row):
        styles = []
        for col in row.index:
            if col == 'Null Hypothesis':  # Dim the first column
                styles.append('color: gray;')
            elif row[col] == 'Correct':
                styles.append('background-color: lightgreen; color: darkgreen; font-weight: bold;')
            elif row[col] == 'Type II Error':
                styles.append('background-color: yellow; color: darkred; font-weight: bold;')
            elif row[col] == 'Type I Error':
                styles.append('background-color: lightpink; color: darkred; font-weight: bold;')
            else:
                styles.append('')  # No styling for other cells
        return styles

    # Apply the styling to the DataFrame
    styled_df = df.style.apply(highlight_cells, axis=1)

    # Display the styled DataFrame in Streamlit
    st.dataframe(styled_df)

    st.markdown(
        r"""
Here is a brief explanation. In linear regression, the *Null Hypothesis* is that **nothing happens**, 
meaning that there is no TRUE impact of a variation of x on the values of y. I mentioned earlier, we don't
know the truth, for we don't have the entire population, we estimate using amostration, and here is where
the erros table comes is:

- When the **Truth** is that Nothing Happens (Null Hypothesis) and we ACCEPT it, it's **correct**;
- If we REJECT the Null Hypothesis when it is True (fake positive), it's an **Type I Error**;
- If we REJECT the Null Hypothesis when it is False (x does impact y), it's **correct**;
- Finally, if we ACCEPT the Null Hypothesis when it's False (fake negative), it's an **Type II Error**;

**Important:** The proper terminology is 'Not-Reject' instead of 'Accept', I used Accept (which is wrong)
to make it easier to understand.


### Assessing the Accuracy of the Model
(This part will be somewhat confusing, I'll try my best to explain clearly)

There is two quantities that are generally attributed to the quality of a linear regression: the *residual
standard error* (RSE) and the $R^2$ statistic. - Equations on CheatSheet.

- **The Residual Standard Error (RSE):**

The RSE provides an estimation of the standard deviation of $\epsilon$ (Figure 3.3, pg. 83), if we 
take a bunch of amostrations with different error term, calculated the regression line for each and 
retreive the average slope and intercept, it will be very close to the true intercept and slope that comes
from the population (considering the error term are normally distributed with mean zero).

The interpretation of the RSE is basically that, even if the true values for the intercept and slope were
known, the prediction would STILL be off by an 'RSE' units, on average (if RSE = 200, the prediction 
would be off by 200 units on average). Is basically the average expected difference between the 
'Observed x Predicted' values. The further the predicted value is from the observed value, the larger will 
the RSE be, indicating that the model doesn't fit the data well.

- **R² Statistic:**

As an economist myself, I do not really take the R² nor the adjusted-R² as of much relevance in my 
alaysis, for the P-value, the slope and the OLS premisses are of greater interest, and the interpretation
of R² might be deceiving. 

But since the book do give it it's credit, I'll explain what it tells. Basically, it is a measurement 
of how much of the variation of 'y' is explained by 'x'. It is calculated using the TSS and RSS:
- TSS: total variance of Y, the total variability *before* the regression is performed;
- RSS: amount of variability left unexplained (residual error) after performing the regression;

When R² is closer to 1, it means that there is a *high proportion of variability in Y that can be explained
using X*, if closer to 0, then is the opposite. For example, a R² of 0,75 would mean that 3/4 of the
variability in Y is explained by the linear regression on X.

Again, this parameter might be misleading, usually used just to confirm already known behaviours on from the 
data, for example: considering we have Y as *Personal Spendings* and X as *Income*. It is expected that
there is a high proportion of variability in Y explained by X, so a R² close to zero would indicate a 
very strange behaviour in our analysis, and would be a good idea to go check the previous steps on the 
regression.

Ps.: note the , considering: $r = Cor(X, Y)$, then $R² = r²$

"""
    )

