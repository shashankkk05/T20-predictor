#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import joblib
import pandas as pd

# Load trained model and encoders
model = joblib.load('t20_model.pkl')
le_team = joblib.load('label_encoder_team.pkl')
le_venue = joblib.load('label_encoder_venue.pkl')
le_winner = joblib.load('label_encoder_winner.pkl')

st.set_page_config(page_title="T20 Match Winner Predictor", layout="centered")
st.title("🏏 T20 International Match Winner Predictor")

# Build lists of options for dropdowns
teams = list(le_team.classes_)
venues = list(le_venue.classes_)

# User input
st.sidebar.header("Match Details")
home_team = st.sidebar.selectbox("Home Team", teams)
away_team = st.sidebar.selectbox("Away Team", teams)
venue = st.sidebar.selectbox("Venue", venues)

# Prevent same team selection
if home_team == away_team:
    st.sidebar.error("Home and Away teams must be different.")

# Predict button
def predict_winner(home, away, venue_val):
    # Encode inputs
    home_enc = int(le_team.transform([home])[0])
    away_enc = int(le_team.transform([away])[0])
    venue_enc = int(le_venue.transform([venue_val])[0])
    # Generate prediction
    pred_enc = model.predict([[home_enc, away_enc, venue_enc]])[0]
    pred_team = le_winner.inverse_transform([pred_enc])[0]
    # Predict probabilities
    probs = model.predict_proba([[home_enc, away_enc, venue_enc]])[0]
    # Map probabilities to teams
    prob_df = pd.DataFrame({
        'Team': le_winner.inverse_transform(model.classes_),
        'Win Probability': probs
    }).sort_values(by='Win Probability', ascending=False)
    return pred_team, prob_df

if st.button("Predict Winner"):
    if home_team != away_team:
        winner, prob_df = predict_winner(home_team, away_team, venue)
        st.subheader(f"Predicted Winner: {winner}")
        st.write("### Win Probabilities")
        st.dataframe(prob_df.reset_index(drop=True))
    else:
        st.warning("Please select two different teams.")

st.markdown("---")
st.info("Built with Streamlit and a Random Forest model. Reload to predict again.")

# To run this app:
# streamlit run app.py

