import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# import the data
df = pd.read_csv('data/Advertising.csv')

# Extract x and y (adding constand for intercept)
x = df[['TV']].values
x = sm.add_constant(x)
y = df[['sales']].values

# apply model
model = sm.OLS(y, x)
result = model.fit()

# check results
# print(result.summary())
print(type(x))
print(type(result))

# set params
parameters = result.params
intercept = parameters[0]
slope = parameters[1]

sns.set_theme(style="darkgrid")
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# Scatterplot
sns.scatterplot(data=df, x='TV', y='sales', color='blue')

# Regression line
df['predicted_sales'] = intercept + slope * df['TV']
ax.plot(df['TV'], df['predicted_sales'], color='red', linewidth=2, label='Regression Line')

# Residuals
for i in range(len(df)):
    ax.plot([df['TV'][i], df['TV'][i]], [df['sales'][i], df['predicted_sales'][i]], color='gray', linewidth=0.4)

# Set axis limits and labels
# ax.set_ylim(0, 30)
# ax.set_xlim(0, 300)
ax.set_xlabel('TV')
ax.set_ylabel('Sales')
ax.legend()
plt.show()