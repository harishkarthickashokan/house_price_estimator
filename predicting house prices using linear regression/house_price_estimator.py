import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("house_data.csv")
print(df.head())

X = df[['Area', 'BHK', 'Age', 'Furnished']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nEnter House Details to Predict Price:")
area = float(input("Enter Area (sqft): "))
bhk = int(input("Enter BHK (Bedrooms): "))
age = int(input("Enter Age of Property (years): "))
furnished = int(input("Furnished? (1=Yes, 0=No): "))

user_input = pd.DataFrame([[area, bhk, age, furnished]],
                          columns=['Area', 'BHK', 'Age', 'Furnished'])

predicted_price = model.predict(user_input)

price = int(round(predicted_price[0]))

print("\nEstimated House Price:","₹", price)
