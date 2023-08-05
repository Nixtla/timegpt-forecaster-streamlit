import os
import requests
from datetime import datetime, timedelta

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from plotly.colors import n_colors

load_dotenv()


def dates_to_dataframe(data, freq, horizon, default, countries):
    start_date = data['ds'].min()
    end_date = data['ds'].max()
    # Generate a list of dates for the entire period, including the horizon
    date_range = pd.date_range(start=start_date, periods=len(data) + horizon, freq=freq)
    df = pd.DataFrame(date_range, columns=['ds'])
    if countries != "":
        countries = countries.split(',')
        for country in countries:
            # Get a list of holidays for that country in the date range
            country_holidays = holidays.CountryHoliday(country, years=df['ds'].dt.year.unique().tolist())
            
            # Create new column in dataframe for that country
            df[country] = df['ds'].apply(lambda x: x in country_holidays).astype(int)
    if default:
        # Frequencies containing day information
        day_freqs = ["D", "B", "H", "T", "S", "L", "U", "N"]
        # Frequencies containing week or month information
        week_month_freqs = ["W", "M", "Q", "A", "Y"]
        if any(freq.startswith(day_freq) for day_freq in day_freqs):
            df['day_of_week'] = df['ds'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
            df['week_number'] = df['ds'].dt.isocalendar().week
            # Convert day_of_week to dummy variables
            day_dummies = pd.get_dummies(df['day_of_week'], prefix='day', drop_first=True).astype(int)
            df = pd.concat([df, day_dummies], axis=1)
            df['month'] = df['ds'].dt.month
            # Convert month to dummy variables
            month_dummies = pd.get_dummies(df['month'], prefix='month').astype(int)
            df = pd.concat([df, month_dummies], axis=1)
            df = df.drop(columns=['month', 'day_of_week'])
        if any(freq.startswith(week_month_freq) for week_month_freq in week_month_freqs):
            #df['week_number'] = df['ds'].dt.isocalendar().week
            df['month'] = df['ds'].dt.month
            # Convert month to dummy variables
            month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True).astype(int)
            df = pd.concat([df, month_dummies], axis=1)
            df = df.drop(columns=['month'])
    return df

def preprocess_exogenous(file, cal_df, horizon):
    if file is not None:
        X_df = pd.read_csv(file)
        required_columns = ["unique_id", "ds"]
        assert all(column in X_df.columns for column in required_columns)
        X_df['ds'] = pd.to_datetime(X_df['ds'])
        X_df['unique_id'] = X_df['unique_id'].astype(str)
        if cal_df is not None:
            X_df = X_df.merge(cal_df)
        X_df_test = X_df.groupby('unique_id').tail(horizon)
        X_df_train = X_df.drop(X_df_test.index)
        return X_df_train, X_df_test
    elif cal_df is not None:
        X_df_test = cal_df.groupby('unique_id').tail(horizon)
        X_df_train = cal_df.drop(X_df_test.index)
        return X_df_train, X_df_test
    else:
        return None, None

def predict_from_api(
        df, horizon, X_df, X_df_future,
        finetune_steps,
        level,
        clean_ex_first,
        freq,
    ):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ['NIXTLA_TOKEN']}"
    }
    input_size = requests.post(
        os.environ["INPUT_SIZE_ENDPOINT"],
        json={"freq": freq}, 
        headers=headers,
    ).json()['data']
    df = df.groupby('unique_id').tail(input_size + horizon)
    df['ds'] = df['ds'].astype(str)
    y = df.to_dict(orient='split', index=False)
    if X_df is not None:
        X_df = X_df.groupby('unique_id').tail(input_size + horizon)
        x = pd.concat([X_df, X_df_future])
        x['ds'] = x['ds'].astype(str)
        x = x.to_dict(orient='split', index=False)
    else:
        x = None
    payload = {
        "y": y,
        "x": x,
        "fh": horizon,
        "level": level,
        "finetune_steps": finetune_steps,
        "clean_ex_first": clean_ex_first,
        "freq": freq,
    }
    
    response = requests.post(
        os.environ["FORECAST_ENDPOINT"], 
        json=payload, 
        headers=headers
    ).json()['data']
    y_hat_df = pd.DataFrame(**response['forecast'])
    if x is not None:
        weights = response['weights_x']
    else:
        weights = None
    return y_hat_df, weights

def perform_forecast(file, file_ex, freq, horizon, 
                     finetune_steps, level, add_default_cal_vars,
                     countries):
    df = pd.read_csv(file)
    df['unique_id'] = df['unique_id'].astype(str)
    required_columns = ["unique_id", "ds", "y"]
    assert all(column in df.columns for column in required_columns)
    df["ds"] = pd.to_datetime(df["ds"])
    if add_default_cal_vars or countries != '':
        cal_df = df.groupby('unique_id').apply(
            lambda df: dates_to_dataframe(df, freq, horizon, add_default_cal_vars, countries)
        ).reset_index().drop(columns='level_1')
    else:
        cal_df = None
    X_df, X_df_future = preprocess_exogenous(file_ex, cal_df, horizon)
    forecast_results, weights = predict_from_api(
        df, horizon, 
        X_df, X_df_future,
        finetune_steps=finetune_steps,
        level=[level],
        clean_ex_first=True,
        freq=freq,
    )
    if X_df is not None:
        weights_df = pd.DataFrame({
            'features': X_df.drop(columns=['unique_id', 'ds']).columns.to_list(),
            'weights': weights
        })
    else:
        weights_df = None
    return df, X_df, X_df_future, forecast_results, weights_df

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def summarize_forecast_results(data, forecast_results, horizon, X_df, X_df_future, weights_df, level):
    st.header("Forecast Results")
    st.subheader("Download your forecasts")

    # download data
    csv = convert_df(forecast_results)
    st.download_button(
       "Get TimeGPT forecasts",
       csv,
       "forecasts-timegpt.csv",
       "text/csv",
       key='download-csv'
    )

    st.subheader("Visualize Time Series")
    # Fetch all unique ids
    uids = data['unique_id'].unique()
    # Create dropdown menu for unique id selection
    selected_uid = st.selectbox('Select a unique_id to view', uids)
    # Prepare the dataframe for the selected unique_id
    input_size = 7 * horizon
    df = data[data['unique_id'] == selected_uid].iloc[-input_size:]
    forecast_df = forecast_results[forecast_results['unique_id'] == selected_uid]#.iloc[-input_size:]
    # Create a plotly figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df['ds'], y=df['y'], mode='lines', 
            line=dict(color='blue'), name='Actual Value'
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'], y=forecast_df['TimeGPT'], mode='lines', 
            line=dict(color='red'), name='TimeGPT Forecast',
        ),
    )
    lo = forecast_df[f'TimeGPT-lo-{level}'].to_list()
    hi = forecast_df[f'TimeGPT-hi-{level}'].to_list()
    ds = forecast_df['ds'].to_list()
    fig.add_trace(
        go.Scatter(
            x=ds + ds[::-1],  # X coordinates for the filled area.
            y=hi + lo[::-1],  # Y coordinates for the filled area.
            fill='toself',  # The area under the trace is filled.
            fillcolor='rgba(0,176,246,0.2)',  # The fill color.
            line_color='rgba(255,255,255,0)',  # The line color.
            #showlegend=False,  # The trace is not added to the legend.
            name='Prediction Interval',
        )
    )
    fig.update_layout(
        title='Time Series '+selected_uid,
        xaxis_title='Date',
        yaxis_title='Value'
    )
    # Show the plot
    st.plotly_chart(fig)

    if X_df is not None:
        st.subheader("Importance of Exogenous Variables")
        # Display the weights and their significance
        st.write(weights_df)
        # Plot the weights for a visual representation
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weights_df['features'],
            y=weights_df['weights'],
            marker_color='skyblue'
        ))
        fig.update_layout(
            title='Weights of Covariates',
            xaxis_title='Covariates',
            yaxis_title='Weights',
            showlegend=False,
            xaxis=dict(tickmode='linear'),
            yaxis=dict(tickmode='linear'),
            bargap=0.2
        )
        st.plotly_chart(fig)
        st.subheader("Plot of Exogenous Variables")
        # Prepare the dataframe for the selected unique_id
        X_df_uid = X_df[X_df['unique_id'] == selected_uid].iloc[-input_size:]
        x_cols = X_df.drop(columns=['unique_id', 'ds']).columns
        color_sequence = n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', len(x_cols) + 1, colortype='rgb')
        # Define subplot titles
        subplot_titles = [f"Time Series {selected_uid}"] + x_cols.to_list()
        # Create subplot figure with titles
        fig = make_subplots(rows=len(x_cols) + 1, cols=1, subplot_titles=subplot_titles)
        for j, col_name in enumerate(subplot_titles, 1):
            if j == 1:
                fig.add_trace(
                    go.Scatter(
                        x=df['ds'], y=df['y'], mode='lines', 
                        line=dict(color='blue'), #name='Actual Value'
                        showlegend=False,
                    ),
                    row=j,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=X_df_uid['ds'], y=X_df_uid[col_name], mode='lines', 
                        line=dict(color=color_sequence[j-2]), #name=f'Exogenous col {col_name}',
                        #legendgroup=legend_group,  # Group all exogenous traces in the legend
                        showlegend=False, #(i == 1)
                    ),
                    row=j,
                    col=1,
                )
    
        # Update yaxes with automargin=True for all rows
        for row in range(1, len(x_cols) + 2):
            fig.update_yaxes(automargin=True, row=row, col=1)
        # Update xaxes globally
        fig.update_xaxes(title_text="Date")
        # Set the global title of the figure
        fig.update_layout(height=200*(len(x_cols) + 1))#, title_text="Time Series and Covariates")
        st.plotly_chart(fig)

def main():
    st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    
    # Function to set the stage state to a specific value.
    def set_state(i):
        st.session_state.stage = i
        
    if 'processed' not in st.session_state:
        st.session_state.processed = {}
    st.title("Time Series Forecasting App")
    st.write("👋 Welcome to Nixtla's forecasting app, your one-stop 🎯 solution for predicting your time series with precision powered by TimeGPT.") 
        
    st.header("Upload Data and Define Horizon")
    
    st.subheader("Data")
    cols_data = st.columns(2)
    example_data_url = 'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity.csv'
    example_data_x_url = 'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/exogenous-vars-electricity.csv'
    with cols_data[0]:
        uploaded_file = st.file_uploader(
            f"Upload your time series data (CSV format) \n[Download Target Example Data]({example_data_url})", 
            type=["csv"]
        )
    with cols_data[1]:
        uploaded_file_ex = st.file_uploader(
            f"Upload your exogenous data (CSV format, Optional) \n[Download Exogenous Variables Example Data]({example_data_x_url})", 
            type=["csv"]
        )

    st.subheader("Forecasting parameters")
    cols_params = st.columns(4)
    with cols_params[0]:
        freq = st.text_input("Define the frequency of your data (see [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases))", value="MS")
    with cols_params[1]:
        horizon = st.number_input("Define forecast horizon (in number of timestamps you want to predict)", value=12)
    with cols_params[2]:
        finetune_steps = st.number_input("Define finetune steps (use zero for zero-shot inference, which is faster)", value=0)
    with cols_params[3]:
        level = st.number_input("Define level for prediction intervals (uncertainty estimation)", min_value=1, max_value=99, value=90)
    st.subheader("Calendar variables")
    col_params_cal = st.columns(2)
    with col_params_cal[0]:
        add_default_cal_vars = st.selectbox("Add default calendar variables", [True, False])
    with col_params_cal[1]:
        countries = st.text_input("Add country holidays (separated by comma, eg UnitedStates,Mexico)", value="")
        
    if st.button("Run Forecast"):
        if uploaded_file is None:
            st.warning("Please upload a CSV file. Using sample data for now.")
            # Delete later
            file = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers_format.csv"
            file_x = None
        else:
            file = uploaded_file
            file_x = uploaded_file_ex
        data, X_df, X_df_future, forecast_results, weights_df = perform_forecast(
            file, file_x, freq, horizon, finetune_steps,
            level, add_default_cal_vars,
            countries,
        )
        # Show results and move to Step 2
        st.success("Forecasting completed!")
        st.session_state.processed["data"] = data
        st.session_state.processed["X_df"] = X_df
        st.session_state.processed["X_df_future"] = X_df_future
        st.session_state.processed["forecast_results"] = forecast_results
        st.session_state.processed["weights_df"] = weights_df
        st.session_state.processed["horizon"] = horizon
        st.session_state.processed["level"] = level
    if st.session_state.processed.get("data", None) is not None:
        summarize_forecast_results(
            st.session_state.processed["data"], 
            st.session_state.processed["forecast_results"], 
            st.session_state.processed["horizon"], 
            st.session_state.processed["X_df"], 
            st.session_state.processed["X_df_future"], 
            st.session_state.processed["weights_df"],
            st.session_state.processed["level"],
        )

if __name__ == "__main__":
    main()

