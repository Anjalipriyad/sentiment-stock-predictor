# backend/app/database/crud.py

from sqlalchemy.orm import Session
from backend.app.database import models

# -----------------------------------------------------------
# Create a new prediction record in the database
# -----------------------------------------------------------
def create_prediction(
    db: Session,
    ticker: str,
    best_model: str,
    predicted_price: float,
    predicted_direction: str,
    direction_metrics: dict,
    price_metrics: dict,
):
    """
    Inserts a prediction result into the database.
    Saves direction metrics from the stacked LSTM model instead of RF.
    """
    lstm_metrics = direction_metrics.get("lstm_stacked", {})

    new_record = models.PredictionResult(
        ticker=ticker,
        best_model=best_model,
        direction_accuracy=float(lstm_metrics.get("accuracy", 0)),
        direction_precision=float(lstm_metrics.get("precision", 0)),
        direction_recall=float(lstm_metrics.get("recall", 0)),
        direction_f1=float(lstm_metrics.get("f1", 0)),
        price_mae=float(price_metrics.get("random_forest", {}).get("MAE", 0)),  # keep price metrics from RF
        price_rmse=float(price_metrics.get("random_forest", {}).get("RMSE", 0)),
        price_r2=float(price_metrics.get("random_forest", {}).get("R2", 0)),
        predicted_price=float(predicted_price),
        predicted_direction=predicted_direction,
        metrics_json={
            "direction": direction_metrics,  # store full metrics JSON
            "price": price_metrics,
        },
    )

    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return new_record


# -----------------------------------------------------------
# Fetch all stored predictions
# -----------------------------------------------------------
def get_all_predictions(db: Session):
    return db.query(models.PredictionResult).order_by(models.PredictionResult.id.desc()).all()


# -----------------------------------------------------------
# Fetch prediction results for a specific ticker
# -----------------------------------------------------------
def get_predictions_by_ticker(db: Session, ticker: str):
    return (
        db.query(models.PredictionResult)
        .filter(models.PredictionResult.ticker == ticker)
        .order_by(models.PredictionResult.id.desc())
        .all()
    )
