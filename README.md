# TimeGPT-Forecaster - Unleashing Time Series Predictions 📈⏲️

Welcome 🙏 to the TimeGPT-Forecaster code repository. This project presents a potent 🔥 tool for performing time series forecasts 💹 with your own data, powered by Nixtla's TimeGPT 💡.

Here is a working version: https://nixtla-timegpt-forecaster.streamlit.app/ (Just click `Run Forecast` to see a live demo!)

## Prerequisites 📚

Before executing 🏃 the project, ensure you have installed the following:

```bash
mamba create -n timegpt-forecaster python=3.10
conda activate timegpt-forecaster
pip install -r requirements.txt
```

## Configuration 🔧

To run this project, several environment variables must be set. These variables include:

- `NIXTLA_TOKEN`: Your Nixtla API key 🔑

Please contact us to secure your API keys.

## Clone the Repository 🔄

To clone the repository, issue the following command:

```bash
git clone https://github.com/Nixtla/timegpt-forecaster.git
```

## Running the Project 🏃‍♀️

After setting the environment variables and installing the dependencies, you can execute the project with the following command:

```bash
streamlit run app.py
```

This command will start a local server, and you can access the web application by navigating to the supplied URL (typically `http://localhost:8501`) in your web browser 🌐.

## How to Use 🛠️

1. On opening the application, upload your time series data (and optional exogenous variables) using the provided interface.
2. Define the frequency of your data, the forecasting horizon and additional variables (such as calendar effects).
3. Click 'Run Forecast' to initiate a forecast based on the uploaded data.
4. If required, you can also adjust various forecasting parameters using the available fields before initiating the forecast.
5. The application will display the forecast results, which can be downloaded for further analysis.

Please bear in mind that some operations may take longer due to the complex calculations involved. Your patience is valued. Revel in the power of time series forecasting! ✨

## Contributing 👥

Pull requests are welcomed. For significant changes, kindly open an issue first to discuss what you would like to alter.

## License 📃

Please refer to the [LICENSE](LICENSE.md) file for specifics.

## Contact 📞

For any queries, feel free to get in touch. We're always ready to assist!
