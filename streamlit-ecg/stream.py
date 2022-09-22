import time
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import altair as alt

import preprocess
import pipeline

load_dotenv()

DELTA = int(os.getenv("DELTA"))
RESAMPLING = int(os.getenv("RESAMPLING"))

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# Load the dataset or build it
try:
    df = preprocess.load_dataset()
except Exception as exce:
    df = preprocess.build_dataset(autosave=True)

# Shuffle it !
#df = df.sample(frac=1)

# Extract the symbols (for the legends)
symbols = df.symbol.unique().tolist()

# Train or load the model
try:
    pipe = pipeline.load()
except Exception as exce:
    pipe = pipeline.build(df, features=["beat"], target="symbol")   

# Set the title and the empty figures
st.title('Heart Beat Analysis')

placeholder = st.empty()
fig_signal = st.empty()
fig_prediction  = st.empty()


# For each beat in the dataset
for cursor in range(len(df)-DELTA):

    # Get this beat and the following ones
    df_window = df.iloc[cursor:cursor+DELTA]

    # Create a vector of beat numbers
    x = np.linspace(cursor, cursor+DELTA, DELTA*RESAMPLING)

    # Create a vector of beat values
    y = np.concatenate(df_window.beat.values).reshape(DELTA, -1)
    
    # Get the first beat of the window
    first_beat = df_window.iloc[0]

    # Make the prediction for the selected beats
    symbol_pred = pipe.predict(y)

    # Extract the true symbols of thoses same beats
    symbol_true = df_window.symbol.values

    # Check if there is an error in those predictions
    prediction_error = np.any(symbol_pred != symbol_true)


    prection_label = "Diagnostic ⏳ OK" if not prediction_error else "Diagnostic ⏳ WRONG"

    with placeholder.container():

        # Create 2 metrics
        kpi1, kpi2 = st.columns(2)

        # Mean value of bpm in the window
        mean_bpm = df_window.bpm.mean()

        # Mean value of bpm in the first beat
        mean_last_bpm = first_beat.bpm.mean()

        # Show those informations
        kpi1.metric(
            label="BPM ⏳",
            value=round(mean_bpm),
            delta=round(mean_bpm - mean_last_bpm),
        )

        # Show the prediciton and the true value of the first beat of the window 
        kpi2.metric(
            label=prection_label,
            value=symbol_pred[-1],
            delta=((symbol_pred[0]==symbol_true[0])-0.5)*2,
        )


        with fig_signal:
            # Build a dataframe
            df_signal = pd.DataFrame.from_dict(
                    {
                        "x" : x,
                        "y" : y.flatten(),
                        "t" : np.repeat(symbol_true, RESAMPLING),
                        "p" : np.repeat(symbol_pred, RESAMPLING),
                    }
                )
            
            # Build a chart with this dataframe
            altair_beat = alt.Chart(df_signal).mark_line().encode(
                    x=alt.X('x:Q', scale=alt.Scale(zero=False, domain=[cursor,cursor+DELTA])),
                    y=alt.Y('y:Q', scale=alt.Scale(zero=False, domain=[-2,2])),
                ).properties(width=100)

            # Plot it
            st.altair_chart(altair_beat, use_container_width=True)

        with fig_prediction:

            # Build a dataframe
            df_prediction = pd.DataFrame.from_dict(
                    {
                        "x" : np.arange(len(symbol_true)),
                        "target" : symbol_true,
                        "predicted" : symbol_pred,
                    }
                )

            # Build a chart with this dataframe
            altair_prediction = alt.Chart(df_prediction).transform_fold(
                ['target', 'predicted',],
                as_=['type', 'value']).mark_bar().encode(
                    x=alt.X('type:N'),
                    y=alt.Y('value:N', scale=alt.Scale(domain=symbols)),
                    column='x:N',
                    color=alt.condition(
                        alt.datum.target != alt.datum.predicted,
                        alt.value("red"),
                        "value:N",
                        scale=alt.Scale(domain=symbols)
                    )
                ).properties(width=150)

            # Plot it
            st.altair_chart(altair_prediction, use_container_width=False)

    # If there is an error, slow down !
    if prediction_error:
        time.sleep(0.75)
    else:
        time.sleep(0.25)