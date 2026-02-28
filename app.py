import streamlit as st
import pandas as pd
import joblib
import os

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="CSAO Recommendation System",
    layout="wide"
)

st.title("🛒 Cart Super Add-On (CSAO) Recommendation System")
st.caption("Session-based, context-aware add-on recommender")

# =============================
# UI-only mappings
# =============================
CATEGORY_MAP = {
    1: "Snacks🥪",
    2: "Beverages🍹",
    3: "Desserts🍨",
    4: "Meals🍛"
}


# =============================
# Load artifacts
# =============================
MODEL_PATH = "csao_xgb_model.pkl"
FEATURES_PATH = "feature_cols.pkl"
DATA_PATH = "final_training_dataset (2).csv"

if not all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH, DATA_PATH]):
    st.error("❌ Required files not found. Please check model, features, or dataset.")
    st.stop()

model = joblib.load(MODEL_PATH)
FEATURE_COLS = joblib.load(FEATURES_PATH)
df = pd.read_csv(DATA_PATH)

st.success("✅ Model, features, and dataset loaded successfully.")

# =============================
# CART UI
# =============================
st.subheader("🍱 Build Your Cart")

if "cart_items" not in st.session_state:
    st.session_state.cart_items = []

item_category = st.selectbox(
    "Select item category",
    options=sorted(CATEGORY_MAP.keys()),
    format_func=lambda x: CATEGORY_MAP[x]
)

if st.button("Add to Cart"):
    # Pick a realistic price from dataset for this category
    price = (
        df[df["item_category"] == item_category]["item_price"]
        .sample(1)
        .values[0]
    )

    st.session_state.cart_items.append({
        "item_category": item_category,
        "item_price": price
    })

    st.success("Item added to cart")
# =============================
# Display Cart
# =============================
st.subheader("🛒 Current Cart")

if st.session_state.cart_items:
    cart_df = pd.DataFrame(st.session_state.cart_items)
    st.dataframe(cart_df, use_container_width=True)
    st.write(f"**Cart Size:** {len(cart_df)}")
    st.write(f"**Cart Value:** ₹{cart_df['item_price'].sum()}")
else:
    st.info("Cart is empty")

# =============================
# RECOMMENDATIONS
# =============================
st.subheader("⭐ Recommended Add-Ons")

if st.session_state.cart_items:
    last_item = st.session_state.cart_items[-1]

    # Sample candidate items (mock catalog)
    candidate_df = df.sample(30, random_state=42).copy()

    # -----------------------------
    # Update cart context
    # -----------------------------
    candidate_df["cart_size"] = len(st.session_state.cart_items)
    candidate_df["cart_total_value"] = cart_df["item_price"].sum()
    candidate_df["last_item_category"] = last_item["item_category"]
    candidate_df["last_item_price"] = last_item["item_price"]

    # -----------------------------
    # 🔧 FORCE REALISTIC CONTEXT
    # -----------------------------
    candidate_df["hour"] = 20                 # dinner time
    candidate_df["weekend"] = 0
    candidate_df["meal_slot_encoded"] = 2     # dinner
    candidate_df["step_number"] = len(st.session_state.cart_items)

    MAX_BUDGET = 500
    candidate_df["budget_utilization"] = (
        candidate_df["cart_total_value"] / MAX_BUDGET
    )
    candidate_df["remaining_budget"] = (
        MAX_BUDGET - candidate_df["cart_total_value"]
    )

    # Drop label if present
    if "label" in candidate_df.columns:
        candidate_df = candidate_df.drop(columns=["label"])

    # -----------------------------
    # Ensure correct feature order
    # -----------------------------
    candidate_df = candidate_df[FEATURE_COLS]

    # -----------------------------
    # Predict probabilities
    # -----------------------------
    candidate_df["score"] = model.predict_proba(candidate_df)[:, 1]

    # -----------------------------
    # Rank & select Top-K
    # -----------------------------
    top_k = candidate_df.sort_values("score", ascending=False).head(5)

    for _, row in top_k.iterrows():
        st.write(
            f"🍽️ **{CATEGORY_MAP.get(row['item_category'], 'Item')}** "
            f"| ₹{int(row['item_price'])} "
            f"| Recommended"
        )

else:

    st.info("Add items to cart to see recommendations.")


