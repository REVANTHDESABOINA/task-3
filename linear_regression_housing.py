import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('/Users/maruthichethan/Desktop/aiml intern/Housing.csv')


# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_cols)],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

# Simple regression plot (area vs price)
X_area = df[['area']]
y_price = df['price']
simple_model = LinearRegression()
simple_model.fit(X_area, y_price)
y_area_pred = simple_model.predict(X_area)

plt.figure(figsize=(10, 6))
plt.scatter(X_area, y_price, color='blue', label='Actual Data')
plt.plot(X_area, y_area_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Area vs Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
