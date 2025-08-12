# ML2.py — Final version (7 features)

import pandas as pd
import joblib

def predict_efficiency(mean_speed, var_speed, mode_speed, num_stops, shift_duration_h, total_mileage_km, idle_time_ratio):
    model = joblib.load("results/model.pkl")
    scaler = joblib.load("results/scaler.pkl")

    input_df = pd.DataFrame([{
        "mean_speed": mean_speed,
        "var_speed": var_speed,
        "mode_speed": mode_speed,
        "num_stops": num_stops,
        "shift_duration_h": shift_duration_h,
        "total_mileage_km": total_mileage_km,
        "idle_time_ratio": idle_time_ratio
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    return prediction

if __name__ == "__main__":
    example_prediction = predict_efficiency(
        mean_speed=8.5,
        var_speed=2.0,
        mode_speed=7.0,
        num_stops=15,
        shift_duration_h=5.5,
        total_mileage_km=45.0,
        idle_time_ratio=0.15
    )
    print(f"🔮 Predicted recapture efficiency: {round(example_prediction, 4)}")
