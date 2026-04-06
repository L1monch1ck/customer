# predict.py
import pickle

# 1️⃣ Загружаем сохранённую модель
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Модель загружена. Введите данные для предсказания.\n")

# 2️⃣ Функция для безопасного ввода числа
def get_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Ошибка: введите число!")

# 3️⃣ Ввод данных пользователем
avg_session = get_float("Avg. Session Length: ")
time_on_app = get_float("Time on App: ")
time_on_website = get_float("Time on Website: ")
length_of_membership = get_float("Length of Membership: ")

# 4️⃣ Предсказание
prediction = model.predict([[avg_session, time_on_app, time_on_website, length_of_membership]])
print(f"\n💰 Предсказанная Yearly Amount Spent: {prediction[0]:.2f}")