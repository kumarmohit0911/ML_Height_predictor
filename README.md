# Height Predictor: My First ML Project

## Overview
This is my first machine learning project where I built a simple **height predictor** using the **linear regression technique**. The dataset contains weight and height measurements, and the goal is to predict height based on weight.

---

## Prerequisites
To run this project, you need the following Python libraries installed:

- `pandas`
- `matplotlib`
- `numpy`
- `seaborn`

Install them using:
```bash
pip install pandas matplotlib numpy seaborn
```

---

## Code Walkthrough

### 1. Importing Necessary Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline 
```

### 2. Loading the Dataset
Load the dataset into a pandas DataFrame and inspect the first few rows.
```python
df = pd.read_csv('height-weight.csv')
df.head()
```

### 3. Visualizing the Data
#### Scatter Plot
A scatter plot is used to project our data points with respect to weight and height.
```python
plt.scatter(df['Weight'], df['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Scatter Plot of Weight vs Height')
plt.show()
```

#### Using Seaborn for Pairwise Visualization
```python
import seaborn as sns
sns.pairplot(df)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()
```

---

## Implementing Linear Regression
Here we use the simple linear regression technique to train a model that predicts height based on weight.

### 4. Splitting the Dataset
Split the dataset into training and testing sets.
```python
from sklearn.model_selection import train_test_split
X = df[['Weight']]
y = df['Height']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Training the Model
Train a linear regression model on the training data.
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### 6. Making Predictions
Use the trained model to make predictions on the test data.
```python
y_pred = model.predict(X_test)
```

### 7. Evaluating the Model
Evaluate the model's performance using metrics like Mean Squared Error (MSE) and R-squared.
```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

---

## Results
- The linear regression model successfully predicts height based on weight.
- Performance metrics:
  - **Mean Squared Error**: `{mse}`
  - **R-squared**: `{r2}`

---

## Dataset
The dataset used in this project is `height-weight.csv`, which includes columns for `Weight` and `Height`.

---

## Future Work
- Explore adding more features to improve the model's accuracy.
- Experiment with other regression techniques like polynomial regression or ridge regression.

---

## Acknowledgements
This project is a stepping stone in my journey to becoming proficient in machine learning. Suggestions for improvement are always welcome!

