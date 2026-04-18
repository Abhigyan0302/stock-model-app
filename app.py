import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stock Model Framework", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('sentgate_dashboard_data.csv')

df = load_data()

# Initialize session state for navigation if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = "System Overview"

# Function to change pages
def move_to_page(page_name):
    st.session_state.page = page_name

# --- System Overview (Home Screen) ---
if st.session_state.page == "System Overview":
    st.title("Stock Model Framework")

    st.write("---")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Accuracy", "52.84%")
    c2.metric("MCC Score", "+0.0657")
    c3.metric("Gate Activation", "19.97%")

    st.write("---")
    # Large button to navigate
    if st.button("View Historical Prediction Audit", use_container_width=True):
        move_to_page("Historical Audit")
        st.rerun()

# --- Historical Audit Page ---
elif st.session_state.page == "Historical Audit":
    if st.button("Back to Home"):
        move_to_page("System Overview")
        st.rerun()

    st.title("Historical Prediction Audit")
    
    st.subheader("Step 1: Filter results by performance")
    filter_choice = st.radio(
        "Choose which dates to display:",
        ["Show All Dates", "Show Successful Dates (Match)", "Show Unsuccessful Dates (Mismatch)"],
        horizontal=True
    )
    
    if filter_choice == "Show Successful Dates (Match)":
        display_df = df[df['final_pred'] == df['actual']]
    elif filter_choice == "Show Unsuccessful Dates (Mismatch)":
        display_df = df[df['final_pred'] != df['actual']]
    else:
        display_df = df

    st.write(f"Displaying {len(display_df)} dates.")

    st.subheader("Step 2: Select a specific day to inspect")
    selected_date = st.selectbox("Select Date:", display_df['Date'].unique())
    row = display_df[display_df['Date'] == selected_date].iloc[0]
    
    st.write("---")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.write("### Prediction")
        st.title("UP" if row['final_pred'] == 1 else "DOWN")
    with res_col2:
        st.write("### Actual Market Move")
        st.title("UP" if row['actual'] == 1 else "DOWN")

    if row['final_pred'] == row['actual']:
        st.success(f"CORRECT CALL: The model accurately predicted the move for {selected_date}.")
    else:
        st.error(f"INCORRECT CALL: Market moved against the model on {selected_date}.")

    st.write("---")
    
    st.subheader("Fusion Breakdown")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("Price Stream (Weighted Vote)")
        st.write(f"**LSTM:** {'UP' if row['lstm_pred'] == 1 else 'DOWN'}")
        st.write(f"**GRU:** {'UP' if row['gru_pred'] == 1 else 'DOWN'}")
        st.write(f"**TFT:** {'UP' if row['tft_pred'] == 1 else 'DOWN'}")
        
    with col2:
        st.info("Sentiment Gate")
        gate_status = "ACTIVE" if row['gate_active'] == 1 else "INACTIVE"
        st.write(f"**Gate Status:** {gate_status}")
        st.write(f"**FinBERT Score:** {row['finbert_score']:.4f}")
        st.write(f"**Phi-3 Score:** {row['phi3_score']:.4f}")

    with col3:
        st.info("Confidence Metrics")
        st.write(f"**Confidence Level:** {row['confidence']:.2%}")
        st.write(f"**Agreement:** {'High' if row['confidence'] > 0.4 else 'Low/Neutral'}")