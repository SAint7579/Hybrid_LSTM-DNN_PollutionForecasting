# Hybrid LSTM + ANN network for AQI forecasting

The project's goal was to address some shortcomings of a general LSTM cell to impove it's performance for predicting the value of Air Quality Index. The solution was a hybrid neural network that splits the input data between an LSTM and a DNN node so that the memory layers inside LSTM do not let the directly affecting factors' older values affect the forecast.<br>

### Model Diagram:
<img src = "https://github.com/SAint7579/ntt_pollution_hybrid_lstm_dnn/blob/master/Images/model.png"></img>

### Dataset:
The dataset used are the AQI and climate attributes of Beijing. The dataset had an hourly timestamp which was aggregated to the mean of dail attributes and the max of PM2.5 value (notebook :1day_mean_max). The attributes in the dataset are:<br>
- PM 2.5
- Temperature
- Moisture
- Air Pressure
- Wind Direction
- Wind Speed 
- Snow
- Rain
### Pollution Variation:
<img src = "https://github.com/SAint7579/ntt_pollution_hybrid_lstm_dnn/blob/master/Images/dataset1.png"></img>

### Forecasting methods:
The forecasting is done using sequence to sequence method where the network takes a seed for the first forecasted value and then feeds on its own predictions to get the next value of the forecast sequence.

### Sequence to Sequence method:
<img src = "https://github.com/SAint7579/ntt_pollution_hybrid_lstm_dnn/blob/master/Images/seq2seq.png"></img>

### Forecast result:
<img src = "https://github.com/SAint7579/ntt_pollution_hybrid_lstm_dnn/blob/master/Images/forecast_14days_lines.png"></img>



