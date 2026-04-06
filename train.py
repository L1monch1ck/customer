# train.py
import os
print("Сохраняю сюда:", os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
data_path = os.path.join(BASE_DIR, "data", "Ecommerce_Customers.csv")

# ✅ ВСЕГДА загружаем данные
data = pd.read_csv(data_path)

# если модели нет — обучаем
if not os.path.exists(model_path):
    print("⚠️ model.pkl не найден, обучаю модель...")

    X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = data['Yearly Amount Spent']

    model = LinearRegression()
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Модель создана автоматически")

# загружаем модель
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 2️⃣ Быстрый анализ данных
print("Информация о данных:")
data.info()
print("\nСтатистика по колонкам:")
print(data.describe(), "\n")

# 3️⃣ Визуализация корреляции с целевой переменной
sns.pairplot(
    data,
    x_vars=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],
    y_vars='Yearly Amount Spent',
    height=5,
    aspect=0.7,
    kind='reg'
)
plt.suptitle("Корреляция признаков с Yearly Amount Spent", y=1.02)
plt.tight_layout()
plt.show()

# 4️⃣ Подготовка данных
X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# 6️⃣ Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}\n")

# 7️⃣ Визуализация фактических vs предсказанных значений
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel("Фактическая сумма Yearly Amount Spent")
plt.ylabel("Предсказанная сумма Yearly Amount Spent")
plt.title("Фактические vs Предсказанные значения")
plt.tight_layout()
plt.show()

# 8️⃣ Сохранение модели
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Модель сохранена в model.pkl")

# 9️⃣ Коэффициенты модели
coefficients = pd.DataFrame({
    'Признак': X.columns,
    'Коэффициент': model.coef_
})
print("Коэффициенты модели:")
print(coefficients)