# backend/app/database/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from backend.app.database.db_setup import Base

# -----------------------------------------------------------
# ORM Model: PredictionResult
# -----------------------------------------------------------
# This table will store:
# - Stock ticker symbol
# - Model type (RandomForest, LSTM, Stacked, etc.)
# - Direction metrics (Accuracy, Precision, Recall, F1)
# - Price metrics (MAE, RMSE, R2)
# - Predicted next-day close price
# - Predicted direction (up/down)
# - Timestamp of prediction
# -----------------------------------------------------------

class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    best_model = Column(String, nullable=False)

    # Direction metrics
    direction_accuracy = Column(Float)
    direction_precision = Column(Float)
    direction_recall = Column(Float)
    direction_f1 = Column(Float)

    # Price metrics
    price_mae = Column(Float)
    price_rmse = Column(Float)
    price_r2 = Column(Float)
    predicted_price = Column(Float)

    # Predicted direction (up/down)
    predicted_direction = Column(String)

    # Optional JSON field to store full metrics dictionary
    metrics_json = Column(JSON)

    # Timestamp for record-keeping
    created_at = Column(DateTime(timezone=True), server_default=func.now())
