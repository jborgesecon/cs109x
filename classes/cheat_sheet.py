import emoji
import streamlit as st


def show_cheatsheets():
    st.header('Cheat Sheet')
    st.markdown(r"""

#### Least Squares Equation:


$$
        Y \approx \beta_0 + \beta_1 X + \epsilon
$$

#### Residual Sum of Squares (RSS): 

$$
RSS = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

#### Residual Standard Error (RSE):
$$
RSE = \sqrt{\frac{1}{n-2} \cdot RSS}
$$
""")

    st.markdown(
        """
        <div style="text-align: center; font-size: 28px;">
            ⬇️
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        r"""

$$
RSE = \sqrt{\frac{1}{n-2} \cdot \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}
$$

#### Total Sum of Squares:
$$
TSS = \sum (y_i - \bar{y})^2
$$
#### R² Statistic:
$$
R^2 = \frac{TSS - RSS}{TSS}
$$

""")