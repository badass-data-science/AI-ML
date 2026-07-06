
## Preparing data for LSTM training

#### Pull forward-filled Forex time series data from the InfluxDB database and save it as a DataFrame

```
prepare-training-and-inference-data.ipynb
```

Alternative, you can run the wrapper for this notebook that iterates through each of the currency pairs and granularities:

```
make-all-training-data.ipynb
```

#### Divide into training, validation, and test sets

```
prepare-ml-ts-data.ipynb
```

## Train an LSTM model

```
python lstm.py
```
