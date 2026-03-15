# ⚡ Sentiment-Based Stock Prediction System

> *"Merging financial data, public sentiment, and AI to see tomorrow’s market today."*

A full-stack **stock prediction platform** that combines **sentiment analysis**, **time-series learning**, and a **hybrid RF + LSTM model** to forecast both **next-day stock prices** and **market direction (UP/DOWN)**.

Built with **FastAPI**, **React**, and **SQLAlchemy**, the system demonstrates the power of combining traditional machine-learning methods with deep-learning architectures for financial forecasting.

---

## 🌟 Key Highlights

- 🧠 **Hybrid RF + LSTM Stacking**  
  A custom ensemble that fuses Random Forest’s interpretability with LSTM’s temporal memory for more robust direction prediction.

- 💸 **Dual Prediction System**  
  - Predicts **Next-Day Closing Price** (via the best regression model).  
  - Predicts **Direction (UP/DOWN)** via the **RF + LSTM stack**.

- 📈 **Interactive Visualization**  
  The frontend renders an intuitive **price chart** that extends historical prices with the predicted next-day point.

- 🧩 **End-to-End Architecture**  
  From data ingestion → ML pipeline → REST API → visualization — all components are seamlessly integrated.

- 💾 **Persistent Storage**  
  Every prediction (ticker, model used, predicted direction & price) is saved for history tracking and analysis.


---

## ⚙️ Technologies Used

### 🧠 Machine Learning
- **Random Forest (RF)** – captures nonlinear patterns in market features.  
- **Long Short-Term Memory (LSTM)** – learns temporal dependencies from past prices.  
- **Stacked Ensemble** – combines both models for final direction prediction.  
- **LightGBM / XGBoost / CatBoost** – evaluated as potential price regression models.  
- **Pandas / NumPy / Scikit-learn / TensorFlow** – ML workflow backbone.

### 💻 Backend
- **FastAPI** – for high-performance async REST endpoints.  
- **SQLAlchemy ORM** – to persist predictions in a relational DB.  
- **Uvicorn** – lightweight ASGI server for deployment.

### 🎨 Frontend
- **React + TypeScript** – modular, fast, and maintainable.  
- **Recharts** – clean visualization of price predictions.  
- **Axios** – communication with backend API.  
- **Vite** – blazing fast frontend build tool.

<img width="902" height="841" alt="image" src="https://github.com/user-attachments/assets/d4806af2-3413-4281-ad89-dbe8176f494e" />
<img width="710" height="872" alt="image" src="https://github.com/user-attachments/assets/40bab3a9-c6e0-47e5-9b0a-7cb3356d1a44" />



