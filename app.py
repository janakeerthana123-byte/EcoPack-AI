from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

app = Flask(__name__)
CORS(app)

# ==========================================================
# LAZY MODEL LOADING (RENDER SAFE)
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
# DATABASE CONNECTION
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
# CREATE TABLE
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


# ==========================================================
# FAST MATERIALS UPLOAD (COPY METHOD)
# ==========================================================
@app.route("/upload-materials")
def upload_materials():
    try:
        df = pd.read_csv("Materials.csv")

        if "material_id" in df.columns:
            df = df.drop(columns=["material_id"])

        conn = get_connection()
        cursor = conn.cursor()

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

        return jsonify({"status": f"{len(df)} materials uploaded successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# RECOMMEND API (FIXED VERSION)
# ==========================================================
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        load_models()

        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        category = data.get("product_category", "")
        weight = float(data.get("product_weight_kg", 1))
        fragility = float(data.get("fragility_level_1_to_10", 5))
        moisture = float(data.get("moisture_sensitivity_0_to_1", 0.5))
        leakage = float(data.get("leakage_risk_0_to_1", 0.5))
        shipping = float(data.get("shipping_stress_index_1_to_10", 5))
        regulatory = float(data.get("regulatory_requirement_level_1_to_5", 3))

        # ---------------- FETCH MATERIALS ----------------
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
            return jsonify({"message": "No materials available."})

        # ---------------- PHYSICAL FILTER ----------------
        df = df[df["weight_capacity_kg"] >= weight * 1.1]
        if df.empty:
            return jsonify({"message": "No materials support this weight."})

        # ---------------- APPLY PRODUCT FEATURES ----------------
        df["product_weight_kg"] = weight
        df["fragility_level_1_to_10"] = fragility
        df["moisture_sensitivity_0_to_1"] = moisture
        df["leakage_risk_0_to_1"] = leakage
        df["shipping_stress_index_1_to_10"] = shipping
        df["regulatory_requirement_level_1_to_5"] = regulatory

        # ---------------- ML PREDICTIONS ----------------
        feature_df = df[MODEL_FEATURES]
        scaled = scaler.transform(feature_df)

        df["predicted_cost"] = cost_model.predict(scaled)
        df["predicted_co2"] = co2_model.predict(scaled)

        norm = MinMaxScaler()

        df["cost_score"] = 1 - norm.fit_transform(df[["predicted_cost"]])
        df["co2_score"] = 1 - norm.fit_transform(df[["predicted_co2"]])

        # ---------------- SUITABILITY MATCHING ----------------

        # Structural match (fragility vs tensile strength)
        tensile_norm = norm.fit_transform(df[["tensile_strength_mpa"]]).flatten()
        df["structural_match"] = 1 - abs(tensile_norm - (fragility / 10))

        # Moisture compatibility
        df["moisture_match"] = 1 - abs(df["moisture_barrier_score"] - moisture)

        # Leakage compatibility
        df["leakage_match"] = 1 - abs(df["leakage_resistance_score"] - leakage)

        # Sustainability pressure (regulation-aware)
        eco_base = (
            df["biodegradability_score"] +
            df["recyclability_percent"] / 100
        ) / 2
        df["eco_match"] = eco_base * (regulatory / 5)

        # ---------------- FINAL INTELLIGENT SCORE ----------------
        df["final_score"] = (
            0.20 * df["cost_score"] +
            0.20 * df["co2_score"] +
            0.20 * df["structural_match"] +
            0.15 * df["moisture_match"] +
            0.15 * df["leakage_match"] +
            0.10 * df["eco_match"]
        )

        df = df.sort_values(by="final_score", ascending=False)

        # ---------------- ENSURE VARIETY ----------------
        selected = []
        used_forms = set()

        for _, row in df.iterrows():
            if row["material_form"] not in used_forms:
                selected.append(row)
                used_forms.add(row["material_form"])
            if len(selected) == 3:
                break

        if len(selected) < 3:
            result_df = df.head(3)
        else:
            result_df = pd.DataFrame(selected)

        return jsonify({
            "status": "success",
            "results": result_df[[
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
    
from flask import render_template

@app.route("/results")
def results_page():
    return render_template("results.html")
    
# ==========================================================
# LOCAL RUN
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)