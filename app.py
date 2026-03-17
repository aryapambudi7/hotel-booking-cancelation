import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #f7f8f5; }

    .header-box {
        background: linear-gradient(135deg, #152f1d 0%, #1e4a2d 100%);
        border-radius: 16px;
        padding: 32px 40px;
        color: white;
        margin-bottom: 28px;
    }
    .header-box h1 { font-size: 28px; font-weight: 700; margin: 0 0 6px 0; color: white; }
    .header-box p  { font-size: 14px; margin: 0; opacity: 0.8; color: white; }

    .section-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #152f1d;
        text-transform: uppercase;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #c8ddc0;
    }

    .result-canceled {
        background: #fef2f2;
        border: 1.5px solid #fca5a5;
        border-radius: 12px;
        padding: 24px 28px;
        text-align: center;
    }
    .result-canceled h2 { color: #dc2626; font-size: 22px; margin: 0 0 4px 0; }
    .result-canceled p  { color: #7f1d1d; font-size: 13px; margin: 0; }

    .result-safe {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 12px;
        padding: 24px 28px;
        text-align: center;
    }
    .result-safe h2 { color: #16a34a; font-size: 22px; margin: 0 0 4px 0; }
    .result-safe p  { color: #14532d; font-size: 13px; margin: 0; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .metric-card .label { font-size: 11px; color: #6b7280; font-weight: 500;
                          text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 22px; font-weight: 700; color: #152f1d; }

    .info-box {
        background: #f0fdf4;
        border-left: 4px solid #152f1d;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #374151;
        margin-bottom: 16px;
    }

    div[data-testid="stButton"] button {
        background: #152f1d !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] button:hover {
        background: #1e4a2d !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model bundle ─────────────────────────────────────
@st.cache_resource
def load_model():
    bundle_path = "hotel_booking_model_bundle.pkl"
    if not os.path.exists(bundle_path):
        st.error("Model file `hotel_booking_model_bundle.pkl` not found. "
                 "Please make sure the file is in the same directory as app.py.")
        st.stop()
    return joblib.load(bundle_path)

bundle          = load_model()
model           = bundle["model"]
scaler          = bundle["scaler"]
ohe             = bundle["ohe"]
ohe_cols        = bundle["ohe_cols"]
feature_columns = bundle["feature_columns"]
deposit_mapping = bundle["deposit_mapping"]

# ── Optimal threshold dari ROC curve (Youden Index) ──────
THRESHOLD = 0.335

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>Hotel Booking Cancellation Prediction</h1>
    <p>Predict whether a hotel booking will be canceled using a Tuned Gradient Boosting model</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────
col_form, col_result = st.columns([2, 1], gap="large")

with col_form:

    # ── Booking Information ──
    st.markdown('<div class="section-title">Booking Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        lead_time = st.number_input(
            "Lead Time (days)", min_value=0, max_value=700, value=30,
            help="Number of days between booking date and arrival date"
        )
        arrival_date_month = st.selectbox(
            "Arrival Month", options=list(range(1, 13)),
            format_func=lambda x: ["January","February","March","April","May","June",
                                    "July","August","September","October","November","December"][x-1]
        )
        adr = st.number_input(
            "Average Daily Rate (ADR)", min_value=0.0, max_value=600.0, value=100.0,
            help="Average room price per night"
        )
    with c2:
        total_guests = st.number_input("Total Guests", min_value=1, max_value=8, value=2)
        total_stay   = st.number_input("Total Nights", min_value=1, max_value=50, value=3)
        room_changed = st.selectbox(
            "Room Changed?", options=[0, 1],
            format_func=lambda x: "Yes (assigned != reserved)" if x == 1 else "No"
        )

    # ── Guest Details ──
    st.markdown('<div class="section-title">Guest Details</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        is_repeated_guest = st.selectbox(
            "Returning Guest?", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        previous_cancellations = st.number_input(
            "Previous Cancellations", min_value=0, max_value=6, value=0
        )
        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Canceled", min_value=0, max_value=70, value=0
        )
    with c4:
        booking_changes = st.number_input(
            "Booking Changes", min_value=0, max_value=21, value=0
        )
        days_in_waiting_list = st.number_input(
            "Days in Waiting List", min_value=0, max_value=150, value=0
        )
        required_car_parking_spaces = st.number_input(
            "Parking Spaces Required", min_value=0, max_value=8, value=0
        )

    total_of_special_requests = st.slider(
        "Number of Special Requests", min_value=0, max_value=5, value=0
    )

    # ── Segmentation & Type ──
    st.markdown('<div class="section-title">Segmentation & Type</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        meal = st.selectbox("Meal Plan", ["BB", "FB", "HB", "SC", "Undefined"])
        market_segment = st.selectbox(
            "Market Segment",
            ["Online TA","Offline TA/TO","Direct","Corporate",
             "Complementary","Groups","Aviation","Undefined"]
        )
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO","Direct","Corporate","GDS","Undefined"]
        )
    with c6:
        reserved_room_type = st.selectbox(
            "Reserved Room Type", ["A","B","C","D","E","F","G","H","L","P"]
        )
        assigned_room_type = st.selectbox(
            "Assigned Room Type", ["A","B","C","D","E","F","G","H","I","K","L","P"]
        )
        customer_type = st.selectbox(
            "Customer Type", ["Transient","Contract","Group","Transient-Party"]
        )
        deposit_type = st.selectbox(
            "Deposit Type", ["No Deposit","Refundable","Non Refund"]
        )
        hotel_segment = st.selectbox("Hotel Type", ["City Hotel","Resort Hotel"])
        country = st.text_input(
            "Country of Origin (code)", value="PRT",
            help="2-3 letter country code, e.g. PRT, GBR, USA"
        )

    predict_btn = st.button("Predict Now", use_container_width=True)

# ── Result panel ─────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        # ── Build input dataframe ──
        input_dict = {
            "lead_time"                      : lead_time,
            "arrival_date_month"             : arrival_date_month,
            "is_repeated_guest"              : is_repeated_guest,
            "previous_cancellations"         : min(previous_cancellations, 6),
            "previous_bookings_not_canceled" : previous_bookings_not_canceled,
            "booking_changes"                : booking_changes,
            "days_in_waiting_list"           : min(days_in_waiting_list, 150),
            "adr"                            : adr,
            "required_car_parking_spaces"    : required_car_parking_spaces,
            "total_of_special_requests"      : total_of_special_requests,
            "total_guests"                   : int(np.clip(total_guests, 1, 8)),
            "total_stay"                     : total_stay,
            "room_changed"                   : room_changed,
            "adr_per_guest"                  : adr / max(total_guests, 1),
            "deposit_type"                   : deposit_mapping.get(deposit_type, 0),
            # Categorical (for OHE)
            "meal"                           : meal,
            "country"                        : country.upper() if country else "Unknown",
            "market_segment"                 : market_segment,
            "distribution_channel"           : distribution_channel,
            "reserved_room_type"             : reserved_room_type,
            "assigned_room_type"             : assigned_room_type,
            "customer_type"                  : customer_type,
            "hotel_segment"                  : hotel_segment,
        }

        input_df = pd.DataFrame([input_dict])

        ohe_input   = ohe.transform(input_df[ohe_cols])
        ohe_df      = pd.DataFrame(ohe_input, columns=ohe.get_feature_names_out(ohe_cols))
        num_input   = input_df.drop(columns=ohe_cols)
        final_input = pd.concat([num_input.reset_index(drop=True),
                                  ohe_df.reset_index(drop=True)], axis=1)

        for col in feature_columns:
            if col not in final_input.columns:
                final_input[col] = 0
        final_input = final_input[feature_columns]

        final_scaled = scaler.transform(final_input)
        proba        = model.predict_proba(final_scaled)[0]
        prob_cancel  = proba[1] * 100
        prob_safe    = proba[0] * 100

        pred = 1 if proba[1] >= THRESHOLD else 0

        # ── Show result ──
        if pred == 1:
            st.markdown(f"""
            <div class="result-canceled">
                <h2>Will Be Canceled</h2>
                <p>Cancellation probability: <strong>{prob_cancel:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <h2>Will Not Be Canceled</h2>
                <p>Not canceled probability: <strong>{prob_safe:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric cards
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Cancel Prob.</div>
                <div class="value" style="color:{'#dc2626' if pred==1 else '#16a34a'}">
                    {prob_cancel:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Safe Prob.</div>
                <div class="value" style="color:#16a34a">{prob_safe:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if prob_cancel >= 70:
            risk       = "High Risk"
            risk_color = "#dc2626"
        elif prob_cancel >= THRESHOLD * 100:   # >= 33.5%
            risk       = "Medium Risk"
            risk_color = "#d97706"
        else:
            risk       = "Low Risk"          
            risk_color = "#16a34a"

        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Risk Level</div>
            <div class="value" style="color:{risk_color}; font-size:16px">{risk}</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:white; border-radius:12px; padding:32px;
                    border:1px dashed #d1d5db; text-align:center; color:#9ca3af;">
            <div style="font-size:14px">Fill in the form on the left<br>
            then click <strong>Predict Now</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model info ──
    st.markdown('<div class="section-title">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Model:</strong> Tuned Gradient Boosting<br>
        <strong>F1-Score:</strong> 0.7951 &nbsp;|&nbsp; <strong>AUC:</strong> 0.9297<br>
        <strong>Optimal Threshold:</strong> 0.335 (Youden Index)
    </div>
    <div class="info-box">
        <strong>Top Predictive Features:</strong><br>
        deposit_type · lead_time · market_segment · total_of_special_requests
    </div>
    """, unsafe_allow_html=True)
