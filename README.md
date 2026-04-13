Set-Content -Path README.md -Value "# Ecommerce Linear Regression Project

- train.py
- predict.py
- web_app.py (Streamlit)

- train.py
- predict.py
- web_app.py
- model.pkl
- data/Ecommerce_Customers.csv
- .gitignore

git clone https://github.com/L1monch1ck/customer.git
cd customer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python train.py
python predict.py
streamlit run web_app.py

Mean Squared Error (MSE)
R² Score
"

git add README.md
git commit -m "add detailed README"
git push