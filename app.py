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
# LOAD MODELS
# ==========================================================
cost_model = joblib.load("cost_model.pkl")
co2_model = joblib.load("co2_model.pkl")
scaler = joblib.load("scaler.pkl")

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

# ==========================================================
# DATABASE CONNECTION
# ==========================================================
import os
import psycopg2

def get_connection():
    return psycopg2.connect(os.environ.get("DATABASE_URL"))

# ==========================================================
# HEALTH CHECK
# ==========================================================
@app.route("/")
def home():
    return jsonify({"status": "EcoPackAI Backend Running"})

# ==========================================================
# RECOMMEND API
# ==========================================================
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
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
        cursor.execute("SELECT * FROM public.materials;")
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

        # ==================================================
        # PHYSICAL FILTER
        # ==================================================
        df = df[df["weight_capacity_kg"] >= weight * 1.1]

        if df.empty:
            return jsonify({"message": "No materials support this weight."})

        # ==================================================
        # CATEGORY PRIORITY (FIXED WARNING HERE)
        # ==================================================
        df["category_priority"] = 0.0  # <-- FIXED (float instead of int)

        if category == "Automotive Parts & Accessories":
            df.loc[df["base_category"] == "Metal", "category_priority"] += 0.5
            df.loc[df["base_category"] == "Plastic", "category_priority"] += 0.2

        elif category == "Food & Beverages":
            is_liquid = moisture > 0.5 or leakage > 0.5
            if is_liquid:
                df.loc[df["base_category"].isin(["Glass","Metal","Plastic"]), "category_priority"] += 0.5
            else:
                df.loc[df["base_category"].isin(["Sustainable","Bioplastic","Paper"]), "category_priority"] += 0.5

        elif category == "Clothing & Textiles":
            df.loc[df["base_category"].isin(["Paper","Plastic","Bioplastic"]), "category_priority"] += 0.4

        elif category == "Electronics & Consumer Goods":
            df.loc[df["base_category"].isin(["Plastic","Bioplastic","Paper","Metal"]), "category_priority"] += 0.3

        elif category == "Luxury & Specialty Items":
            df.loc[df["base_category"].isin(["Glass","Plastic","Metal","Sustainable"]), "category_priority"] += 0.4

        elif category == "Agriculture & Raw Materials":
            df.loc[df["base_category"].isin(["Paper","Sustainable"]), "category_priority"] += 0.4

        elif category == "Cosmetics":
            df.loc[df["base_category"].isin(["Plastic","Glass","Bioplastic"]), "category_priority"] += 0.4

        # ==================================================
        # ML FEATURE PREPARATION
        # ==================================================
        feature_df = pd.DataFrame()

        feature_df["product_weight_kg"] = weight
        feature_df["fragility_level_1_to_10"] = fragility
        feature_df["moisture_sensitivity_0_to_1"] = moisture
        feature_df["leakage_risk_0_to_1"] = leakage
        feature_df["shipping_stress_index_1_to_10"] = shipping
        feature_df["regulatory_requirement_level_1_to_5"] = regulatory

        for col in columns[3:]:
            feature_df[col] = df[col]

        feature_df = feature_df[MODEL_FEATURES]
        scaled = scaler.transform(feature_df)

        df["predicted_cost"] = cost_model.predict(scaled).clip(10, 250)
        df["predicted_co2"] = co2_model.predict(scaled).clip(0.1, 120)

        # ==================================================
        # DYNAMIC SCORING
        # ==================================================
        norm = MinMaxScaler()

        df["cost_score"] = 1 - norm.fit_transform(df[["predicted_cost"]])
        df["co2_score"] = 1 - norm.fit_transform(df[["predicted_co2"]])

        df["load_ratio"] = (df["weight_capacity_kg"] / weight).clip(upper=8)

        df["structure_score"] = (
            norm.fit_transform(df[["tensile_strength_mpa"]]) *
            norm.fit_transform(df[["load_ratio"]])
        )

        df["moisture_score"] = 1 - abs(df["moisture_barrier_score"] - moisture)
        df["leakage_score"] = 1 - abs(df["leakage_resistance_score"] - leakage)

        df["sustain_score"] = (
            df["biodegradability_score"] +
            df["recyclability_percent"] / 100
        ) / 2

        df["final_score"] = (
            0.18 * df["cost_score"] +
            0.18 * df["co2_score"] +
            0.18 * df["structure_score"] +
            0.12 * df["moisture_score"] +
            0.12 * df["leakage_score"] +
            0.12 * df["sustain_score"] +
            0.20 * df["category_priority"]
        ).clip(0, 1)

        df = df.sort_values(by="final_score", ascending=False)

        # ==================================================
        # ENSURE VARIETY
        # ==================================================
        selected = []
        used_forms = set()

        for _, row in df.iterrows():
            if row["material_form"] not in used_forms:
                selected.append(row)
                used_forms.add(row["material_form"])
            if len(selected) == 3:
                break

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

        return {"status": "Table created successfully"}

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    app.run(debug=True)