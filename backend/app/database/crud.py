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
    predicted_direction: str,  # ✅ added
    direction_metrics: dict,
    price_metrics: dict,
):
    """
    Inserts a prediction result into the database.
    """
    new_record = models.PredictionResult(
        ticker=ticker,
        best_model=best_model,
        direction_accuracy=float(direction_metrics.get("random_forest", {}).get("accuracy", 0)),
        direction_precision=float(direction_metrics.get("random_forest", {}).get("precision", 0)),
        direction_recall=float(direction_metrics.get("random_forest", {}).get("recall", 0)),
        direction_f1=float(direction_metrics.get("random_forest", {}).get("f1", 0)),
        price_mae=float(price_metrics.get("random_forest", {}).get("MAE", 0)),
        price_rmse=float(price_metrics.get("random_forest", {}).get("RMSE", 0)),
        price_r2=float(price_metrics.get("random_forest", {}).get("R2", 0)),
        predicted_price=float(predicted_price),
        predicted_direction=predicted_direction,
    


        metrics_json={
            "direction": direction_metrics,
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
