# backend/app/routers/prediction.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.db_setup import get_db
from backend.app.database import crud
from scripts.train_and_evaluate import run_pipeline_stacked  # your training & evaluation pipeline

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ------------------------------------------------------
# 1️⃣ Run prediction and save results to DB
# ------------------------------------------------------
@router.get("/{ticker}")
def predict_stock(ticker: str, db: Session = Depends(get_db)):
    try:
        print(f"📊 Running prediction for {ticker}...")
        results = run_pipeline_stacked(ticker)  # returns dict with metrics + predictions

        if not results or "predicted_price" not in results or "predicted_direction" not in results:
            raise HTTPException(status_code=400, detail="Prediction failed or invalid response.")

        # ✅ Store prediction in DB using stacked model's direction
        crud.create_prediction(
            db=db,
            ticker=ticker.upper(),
            best_model=results.get("best_model"),
            predicted_price=float(results.get("predicted_price")),
            predicted_direction=results.get("predicted_direction"),  # ✅ stacked direction
            direction_metrics=results.get("direction_metrics", {}),
            price_metrics=results.get("price_metrics", {})
        )

        return {
    "ticker": ticker,
    "model": "stacked",
    "predicted_price": results["predicted_price"],
    "predicted_direction": results["predicted_direction"],
    "metrics": {
        "price": results["price_metrics"]["stacked"],
        "direction": results["direction_metrics"]["stacked"]
    },
    "message": "✅ Prediction saved successfully."
}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------
# 2️⃣ Retrieve prediction history from DB
# ------------------------------------------------------
@router.get("/history/{ticker}")
def get_prediction_history(ticker: str, db: Session = Depends(get_db)):
    history = crud.get_predictions_by_ticker(db=db, ticker=ticker)

    if not history:
        raise HTTPException(status_code=404, detail="No predictions found for this ticker.")

    return {
        "ticker": ticker,
        "records": [
            {
                "id": record.id,
                "best_model": record.best_model,
                "predicted_price": record.predicted_price,
                "predicted_direction": record.predicted_direction,  # ✅ stacked direction
                "created_at": record.created_at,
                "direction_accuracy": record.direction_accuracy,
                "price_mae": record.price_mae,
            }
            for record in history
        ],
    }
