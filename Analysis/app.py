import streamlit as st
import pandas as pd
import joblib

from categories import (
    TYPE_OPTIONS,
    CATEGORY_OPTIONS,
    CUSTOMER_COUNTRY_OPTIONS,
    CUSTOMER_SEGMENT_OPTIONS,
    DEPARTMENT_OPTIONS,
    MARKET_OPTIONS,
    ORDER_REGION_OPTIONS,
    SHIPPING_MODE_OPTIONS
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="📦",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f5f7ff 0%,
            #eef2ff 50%,
            #f8faff 100%
        );
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    h1 {
        color: #312e81 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    /* Headings */
    h2 {
        color: #3730a3 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #4338ca !important;
        font-weight: 700 !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 1px solid #dbe2f0 !important;
        background: white !important;
    }

    /* Number inputs */
    div[data-testid="stNumberInput"] input {
        border-radius: 10px !important;
        border: 1px solid #dbe2f0 !important;
        background: white !important;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 12px;
        border: none;

        background: linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

        color: white;
        font-size: 17px;
        font-weight: 700;

        box-shadow:
            0 8px 20px rgba(79,70,229,0.25);

        transition: 0.2s;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 12px 25px rgba(79,70,229,0.35);

        color: white;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;

        box-shadow:
            0 5px 18px rgba(0,0,0,0.06);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

preprocessor = joblib.load(
    "supply_chain_preprocessor.pkl"
)

xgb_model = joblib.load(
    "supply_chain_xgboost.pkl"
)


# =========================================================
# LOAD ORDER COUNTRIES
# =========================================================

with open(
    "order_countries.txt",
    "r",
    encoding="utf-8"
) as f:

    ORDER_COUNTRY_OPTIONS = [
        line.strip("\n")
        for line in f
    ]


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_delivery_risk(
    order_data,
    threshold=0.50
):

    new_order = pd.DataFrame(
        [order_data]
    )

    new_order_processed = (
        preprocessor.transform(new_order)
    )

    late_probability = (
        xgb_model
        .predict_proba(
            new_order_processed
        )[0, 1]
    )

    prediction = int(
        late_probability >= threshold
    )

    return late_probability, prediction


# =========================================================
# MAIN TITLE
# =========================================================

st.title(
    "📦 Late Delivery Risk Prediction"
)

st.caption(
    "Predict potentially delayed orders using machine learning "
    "and supply chain data."
)


# =========================================================
# MODEL BADGE
# =========================================================

st.info(
    "📊 Predictive Analytics • XGBoost Machine Learning Model"
)


# =========================================================
# PROJECT DESCRIPTION
# =========================================================

st.subheader(
    "🎯 About the Prediction System"
)

st.write(
    "Enter the customer, product, order and shipping details "
    "below. The trained XGBoost model estimates the probability "
    "of late delivery."
)


st.divider()


# =========================================================
# ORDER INFORMATION
# =========================================================

st.header(
    "🛒 Order Information"
)

st.caption(
    "Provide the important details about the order."
)


# =========================================================
# TWO COLUMN INPUTS
# =========================================================

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    transaction_type = st.selectbox(
        "Transaction Type",
        TYPE_OPTIONS
    )

    category_name = st.selectbox(
        "Category Name",
        CATEGORY_OPTIONS
    )

    customer_country = st.selectbox(
        "Customer Country",
        CUSTOMER_COUNTRY_OPTIONS
    )

    customer_segment = st.selectbox(
        "Customer Segment",
        CUSTOMER_SEGMENT_OPTIONS
    )

    department_name = st.selectbox(
        "Department Name",
        DEPARTMENT_OPTIONS
    )

    market = st.selectbox(
        "Market",
        MARKET_OPTIONS
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    order_country = st.selectbox(
        "Order Country",
        ORDER_COUNTRY_OPTIONS
    )

    order_region = st.selectbox(
        "Order Region",
        ORDER_REGION_OPTIONS
    )

    shipping_mode = st.selectbox(
        "Shipping Mode",
        SHIPPING_MODE_OPTIONS
    )

    order_month = st.number_input(
        "Order Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )

    order_dayofweek = st.number_input(
    "Order Day of Week (0 = Monday, 6 = Sunday)",
    min_value=0,
    max_value=6,
    value=2,
    step=1
)

# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Delivery Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION RESULT
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # INTERNAL VALUES
    # -----------------------------------------------------
    # These fields are required by the existing trained
    # model but are not shown to the user.
    #
    # The model itself has NOT been retrained.
    # -----------------------------------------------------

    order_item_discount = 20.0
    order_item_discount_rate = 0.10
    order_item_product_price = 150.0
    order_item_quantity = 2
    sales = 270.0
    product_price = 150.0


    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    order_data = {

        "type":
            transaction_type,

        "category_name":
            category_name,

        "customer_country":
            customer_country,

        "customer_segment":
            customer_segment,

        "department_name":
            department_name,

        "market":
            market,

        "order_country":
            order_country,

        "order_region":
            order_region,

        "shipping_mode":
            shipping_mode,

        "order_item_discount":
            order_item_discount,

        "order_item_discount_rate":
            order_item_discount_rate,

        "order_item_product_price":
            order_item_product_price,

        "order_item_quantity":
            order_item_quantity,

        "sales":
            sales,

        "product_price":
            product_price,

        "order_month":
            order_month,

        "order_dayofweek":
            order_dayofweek
    }


    # =====================================================
    # PREDICTION
    # =====================================================

    late_probability, prediction = (
        predict_delivery_risk(
            order_data
        )
    )

    probability_percentage = (
        late_probability * 100
    )


    # =====================================================
    # RESULT HEADING
    # =====================================================

    st.divider()

    st.header(
        "📊 Prediction Result"
    )

    st.caption(
        "Result generated by the trained XGBoost model."
    )


    # =====================================================
    # METRICS
    # =====================================================

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Late Delivery Probability",
            f"{probability_percentage:.2f}%"
        )


    with result_col2:

        if prediction == 1:

            st.metric(
                "Risk Classification",
                "HIGH RISK"
            )

        else:

            st.metric(
                "Risk Classification",
                "LOW RISK"
            )


    # =====================================================
    # RISK MESSAGE
    # =====================================================

    if prediction == 1:

        st.error(
            f"🔴 HIGH RISK — The model predicts a "
            f"{probability_percentage:.2f}% probability "
            f"of late delivery."
        )

    else:

        st.success(
            f"🟢 LOW RISK — The model predicts a "
            f"{probability_percentage:.2f}% probability "
            f"of late delivery."
        )


    # =====================================================
    # MODEL DETAILS
    # =====================================================

    st.subheader(
        "🤖 Model Details"
    )

    detail_col1, detail_col2, detail_col3 = st.columns(3)


    with detail_col1:

        st.metric(
            "Algorithm",
            "XGBoost"
        )


    with detail_col2:

        st.metric(
            "Threshold",
            "50%"
        )


    with detail_col3:

        st.metric(
            "Output",
            "Risk Probability"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📦 Late Delivery Risk Prediction"
)

st.caption(
    "Built with Python • XGBoost • Streamlit"
)