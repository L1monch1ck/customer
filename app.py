# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Настройка страницы
st.set_page_config(page_title="Ecommerce Linear Regression", layout="wide")
st.title("📊 Ecommerce Linear Regression Web Interface")

# 1️⃣ Загрузка данных
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "Ecommerce_Customers.csv")
data = pd.read_csv(data_path)

st.subheader("Превью данных")
st.dataframe(data.head(10))

# 2️⃣ Показываем статистику
st.subheader("Статистика по колонкам")
st.write(data.describe())

# 3️⃣ Показываем график корреляции
st.subheader("Корреляция признаков с Yearly Amount Spent")
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=data, x="Length of Membership", y="Yearly Amount Spent", ax=ax)
sns.regplot(data=data, x="Length of Membership", y="Yearly Amount Spent", ax=ax, scatter=False, color="red")
st.pyplot(fig)

# 4️⃣ Загружаем модель
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 5️⃣ Выводим метрики модели
X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']
y_pred = model.predict(X)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.subheader("Метрики модели")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# 6️⃣ Форма для предсказания
st.subheader("Введите данные для предсказания")
with st.form("prediction_form"):
    avg_session = st.number_input("Avg. Session Length", min_value=0.0, step=0.1)
    time_on_app = st.number_input("Time on App", min_value=0.0, step=0.1)
    time_on_website = st.number_input("Time on Website", min_value=0.0, step=0.1)
    length_of_membership = st.number_input("Length of Membership", min_value=0.0, step=0.1)
    submit = st.form_submit_button("Predict")

if submit:
    prediction = model.predict([[avg_session, time_on_app, time_on_website, length_of_membership]])
    st.success(f"💰 Предсказанная Yearly Amount Spent: {prediction[0]:.2f}")