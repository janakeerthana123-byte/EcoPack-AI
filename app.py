from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# ==========================================================
# LAZY MODEL LOADING (PREVENTS RENDER CRASH)
# ==========================================================
cost_model = None
co2_model = None
scaler = None

MODEL_FEATURES = [
    "product_weight_kg",
    "fragility_level_1_to_10",
    "moisture_sensitivity_0_to_1",
    "leakage_risk_0_to_1",
    "shipping_stress_index_1_to_10",
    "regulatory_requirement_level_1_to_5",
    "tensile_strength_mpa",
    "thickness_mm",
    "weight_capacity_kg",
    "moisture_barrier_score",
    "leakage_resistance_score",
    "biodegradability_score",
    "recyclability_percent",
    "cost_per_kg_inr",
    "co2_kg_per_kg"
]

def load_models():
    global cost_model, co2_model, scaler
    if cost_model is None:
        cost_model = joblib.load("cost_model.pkl")
        co2_model = joblib.load("co2_model.pkl")
        scaler = joblib.load("scaler.pkl")


# ==========================================================
# DATABASE CONNECTION (RENDER SAFE)
# ==========================================================
def get_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port=os.environ.get("DB_PORT"),
        sslmode="require"
    )


# ==========================================================
# HEALTH CHECK
# ==========================================================
@app.route("/")
def home():
    return jsonify({"status": "EcoPackAI Backend Running"})


# ==========================================================
# CREATE TABLE ROUTE
# ==========================================================
@app.route("/create-table")
def create_table():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            material_id SERIAL PRIMARY KEY,
            base_category VARCHAR(100),
            material_form VARCHAR(100),
            tensile_strength_mpa FLOAT,
            thickness_mm FLOAT,
            weight_capacity_kg FLOAT,
            moisture_barrier_score FLOAT,
            leakage_resistance_score FLOAT,
            biodegradability_score FLOAT,
            co2_kg_per_kg FLOAT,
            recyclability_percent FLOAT,
            cost_per_kg_inr FLOAT
        );
        """)

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "Table created successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
from io import StringIO

@app.route("/upload-materials")
def upload_materials():
    try:
        df = pd.read_csv("Materials.csv")

        # ❗ Remove material_id column if it exists
        if "material_id" in df.columns:
            df = df.drop(columns=["material_id"])

        conn = get_connection()
        cursor = conn.cursor()

        # Optional: Clear table before inserting
        cursor.execute("TRUNCATE TABLE materials;")

        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        cursor.copy_expert("""
            COPY materials (
                base_category,
                material_form,
                tensile_strength_mpa,
                thickness_mm,
                weight_capacity_kg,
                moisture_barrier_score,
                leakage_resistance_score,
                biodegradability_score,
                co2_kg_per_kg,
                recyclability_percent,
                cost_per_kg_inr
            )
            FROM STDIN WITH CSV
        """, buffer)

        conn.commit()
        cursor.close()
        conn.close()

        return {"status": f"{len(df)} materials uploaded successfully"}

    except Exception as e:
        return {"error": str(e)}

# ==========================================================
# RECOMMEND API
# ==========================================================
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        load_models()

        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        category = data.get("product_category")
        weight = float(data.get("product_weight_kg", 1))
        fragility = float(data.get("fragility_level_1_to_10", 5))
        moisture = float(data.get("moisture_sensitivity_0_to_1", 0.5))
        leakage = float(data.get("leakage_risk_0_to_1", 0.5))
        shipping = float(data.get("shipping_stress_index_1_to_10", 5))
        regulatory = float(data.get("regulatory_requirement_level_1_to_5", 3))

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM materials;")
        materials = cursor.fetchall()

        columns = [
            "material_id","base_category","material_form",
            "tensile_strength_mpa","thickness_mm",
            "weight_capacity_kg","moisture_barrier_score",
            "leakage_resistance_score","biodegradability_score",
            "co2_kg_per_kg","recyclability_percent",
            "cost_per_kg_inr"
        ]

        df = pd.DataFrame(materials, columns=columns)
        cursor.close()
        conn.close()

        if df.empty:
            return jsonify({"message": "No materials available in database."})

        df = df[df["weight_capacity_kg"] >= weight * 1.1]

        if df.empty:
            return jsonify({"message": "No materials support this weight."})

        df["category_priority"] = 0.0

        if category == "Electronics & Consumer Goods":
            df.loc[df["base_category"].isin(["Plastic","Bioplastic","Paper","Metal"]), "category_priority"] += 0.3

        feature_df = pd.DataFrame({
            "product_weight_kg": weight,
            "fragility_level_1_to_10": fragility,
            "moisture_sensitivity_0_to_1": moisture,
            "leakage_risk_0_to_1": leakage,
            "shipping_stress_index_1_to_10": shipping,
            "regulatory_requirement_level_1_to_5": regulatory
        })

        for col in columns[3:]:
            feature_df[col] = df[col]

        feature_df = feature_df[MODEL_FEATURES]
        scaled = scaler.transform(feature_df)

        df["predicted_cost"] = cost_model.predict(scaled).clip(10, 250)
        df["predicted_co2"] = co2_model.predict(scaled).clip(0.1, 120)

        norm = MinMaxScaler()
        df["cost_score"] = 1 - norm.fit_transform(df[["predicted_cost"]])
        df["co2_score"] = 1 - norm.fit_transform(df[["predicted_co2"]])

        df["final_score"] = (
            0.4 * df["cost_score"] +
            0.4 * df["co2_score"] +
            0.2 * df["category_priority"]
        ).clip(0, 1)

        df = df.sort_values(by="final_score", ascending=False).head(3)

        return jsonify({
            "status": "success",
            "results": df[[
                "material_id",
                "base_category",
                "material_form",
                "predicted_cost",
                "predicted_co2",
                "final_score"
            ]].to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# RUN LOCALLY
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)