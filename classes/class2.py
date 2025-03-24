import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def show_class2():
    st.markdown(
        """
## Multiple Linear Regression (ch. 3, 7)

As we saw earlier, we can check the relation between TV and Sales, on the Advertising data,
but we also have other variables that could explain a variation on Sales. Better than run 
three separate Simple Linear Regressions, we could agregate them and make a single response 
(y) considering all the relevant predictors (x1, x2, ..., xn).

In this case, the resulting graph won't be a 2D chart, but rather different dimensions and 
shapes, a regression with 2 regressors would result in a 3D plane. 

[*there will eventually be a graph here*]

"""
    )
    ## There will eventually be a 3D graph here
    # df = pd.read_csv('data/Advertising.csv')
    # df['intercept'] = np.ones(len(df))
    # x = df[['intercept', 'TV', 'radio']]
    # y = df['sales']
    # model = sm.OLS(y, x)
    # result = model.fit()

    # # Calculate predicted sales using statsmodels
    # df['predicted_sales'] = result.predict(x)

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatterplot customization
    # scatter = ax.scatter(df['TV'], df['radio'], df['sales'], c=df['sales'], cmap='coolwarm', s=30, alpha=0.8)

    # # Plane customization
    # tv_range = np.linspace(df['TV'].min(), df['TV'].max(), 30)
    # radio_range = np.linspace(df['radio'].min(), df['radio'].max(), 30)
    # tv_grid, radio_grid = np.meshgrid(tv_range, radio_range)

    # grid_data = pd.DataFrame({'TV': tv_grid.flatten(), 'radio': radio_grid.flatten()})
    # grid_data['intercept'] = 1
    # grid_data = grid_data[['intercept', 'TV', 'radio']]
    # plane = result.predict(grid_data).values.reshape(tv_grid.shape)

    # ax.plot_surface(tv_grid, radio_grid, plane, alpha=0.4, color='skyblue')

    # # Residual lines (optional, can be removed for cleaner look)
    # for i in range(len(df)):
    #     ax.plot([df['TV'][i], df['TV'][i]], [df['radio'][i], df['radio'][i]], [df['sales'][i], df['predicted_sales'][i]], color='gray', linewidth=0.3)

    # # Axis labels and title
    # ax.set_xlabel('TV', fontsize=12)
    # ax.set_ylabel('radio', fontsize=12)
    # ax.set_zlabel('Sales', fontsize=12)
    # ax.set_title('3D Regression Plane', fontsize=14)

    # # Colorbar
    # fig.colorbar(scatter, ax=ax, label='Sales')

    # # Adjust view angle for better perspective
    # ax.view_init(30, 45)  # Adjust these angles as needed

    # plt.tight_layout() #prevents labels from being cut off.

    # st.pyplot(fig.figure)
    st.markdown(
        """

"""
    )
    return