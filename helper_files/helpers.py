import datetime
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random
import json
import numpy as np
from bisect import bisect_left
import os

with open('config.json') as json_file:
    config = json.load(json_file)

def get_model(r2_client, model_name):
    models = r2_client.get_models()
    print("Loaded", model_name, "Model")
    return next((model for model in models if model["name"] == model_name), None)

def get_charts(r2_client, collection_id, names, interval):
    charts = r2_client.get_charts(collection_id)
    selected_charts = []
    for name in names:
        for chart in charts:
            if chart["name"] == name and chart["interval"] == interval:
                selected_charts.append(chart)
                break
    chart_ids = [chart["id"] for chart in selected_charts]
    print("Loaded Charts: ", chart_ids, '\n')
    return selected_charts, chart_ids

def get_ticker_info(r2_client, chart_collection_id, chart_names, intervals):
    for interval in intervals:
        selected_charts, chart_ids = get_charts(r2_client, chart_collection_id, chart_names, interval)
        interval_df = pd.DataFrame(index=chart_names)
        interval_df["chart_id"] = chart_ids
        all_bars = [r2_client.get_chart(chart_id)["bars"] for chart_id in chart_ids]
        interval_df["bars"] = [len(bars) for bars in all_bars]
        interval_df["start_datetime"] = [bars[0]["datetime"] for bars in all_bars]
        interval_df["end_datetime"] = [bars[-1]["datetime"] for bars in all_bars]
        interval_df.to_csv("results/"+str(interval)+"_ticker_info.csv", index_label="ticker")

def load_ticker_info(intervals, chart_names):
    print("Loaded Ticker Information for Intervals", intervals)
    return [pd.read_csv('results/'+str(interval)+'_ticker_info.csv', index_col=0).loc[chart_names] for interval in intervals]
    
def get_backtesting_parameters(start, end, validation_period_ratio = 0, validation = False, print_output = True, end_trading_time = ""):
    if start is not None and end is not None:
        validation_start = start + (1-validation_period_ratio) * (end - start)
    backtesting_parameters = {
        "start_date": None if start is None else validation_start.strftime("%Y-%m-%dT%H:%M:%S") if validation else start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_date": None if end is None else end.strftime("%Y-%m-%dT%H:%M:%S") if validation else validation_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "close_open_trades": config["close_open_trades"] if "close_open_trades" in config else False,
        "trade_fee": config["trade_fee"],
        "financing_fee": config["financing_fee"],
        "open_trade_limit": config["open_trade_limit"],
        "update_charts": config["update_charts"] if "update_charts" in config else False,
        "end_of_trading_day": config["end_of_trading_day"] if "end_of_trading_day" in config else "",
        "start_trading_time": config["start_trading_time"] if "start_trading_time" in config else "",
        "end_trading_time": end_trading_time
    }
    print("Backtesting Parameters:", backtesting_parameters, "\n") if print_output else None
    return backtesting_parameters

def get_random_strategy_parameters(interval, all_intervals, print_output = True, custom_signals = []):

    strategy_parameters = {}
    if "pattern_threshold_min" and "pattern_threshold_max" in config:
        strategy_parameters["pattern_threshold"] = random.uniform(config["pattern_threshold_min"], config["pattern_threshold_max"])
    if "gradient_threshold_min" and "gradient_threshold_min" in config:
        strategy_parameters["gradient_threshold"] = random.uniform(config["gradient_threshold_min"], config["gradient_threshold_min"])
    if "sl_n_bars_min" and "sl_n_bars_max" in config:
        strategy_parameters["sl_n_bars"] = int(round(random.uniform(config["sl_n_bars_min"], config["sl_n_bars_max"])))
    if "tp_n_bars_min" and "tp_n_bars_max" in config:
        strategy_parameters["tp_n_bars"] = int(round(random.uniform(config["tp_n_bars_min"], config["tp_n_bars_max"])))
    if "tpfrac_min" and "tpfrac_max" in config:
        strategy_parameters["tpfrac"] = random.uniform(config["tpfrac_min"], config["tpfrac_max"])
    if "open_limit_min" and "open_limit_min" in config:
        strategy_parameters["open_limit"] = int(round(random.uniform(config["open_limit_min"], config["open_limit_max"])))
    if "min_risk_reward_min" and "min_risk_reward_max" in config:
        strategy_parameters["min_risk_reward"] = random.uniform(config["min_risk_reward_min"], config["min_risk_reward_max"])
    if "entry_bar_position_threshold_min" and "entry_bar_position_threshold_max" in config:
        strategy_parameters["entry_bar_position_threshold"] = random.uniform(config["entry_bar_position_threshold_min"], config["entry_bar_position_threshold_max"])
    if "limit_order_expiry_min" and "limit_order_expiry_max" in config:
        strategy_parameters["limit_order_expiry"] = int(round(random.uniform(config["limit_order_expiry_min"], config["limit_order_expiry_max"])))
    if "mean_reversion_or_breakout" in config:
        strategy_parameters["mean_reversion_or_breakout"] = config["mean_reversion_or_breakout"] if config["mean_reversion_or_breakout"] != "both" else random.sample(["breakout", "mean_reversion"], 1)[0]
    if "use_tp" in config:
        strategy_parameters["use_tp"] = config["use_tp"]
    if "indicator_as_signal" in config:
        strategy_parameters["indicator_as_signal"] = config["indicator_as_signal"]


    indicators = []
    config_indicators = config["indicators"] if "indicators" in config else []
    for i in range(len(config_indicators)):
        optimisation_indicator = config["indicators"][i]
        indicator = {}
        if "name" in optimisation_indicator:
            indicator["name"] = random.sample(["rsi", "adx", "stochastic", "macd"], 1)[0] if config["indicators"][i]["name"] == "all" else config["indicators"][i]["name"]
        if "interval_min" and "interval_max" in optimisation_indicator:
            indicator["interval"] = random.sample(all_intervals[all_intervals.index(config["indicators"][i]["interval_min"]) : all_intervals.index(config["indicators"][i]["interval_max"]) + 1], 1)[0]
        if "entry_value_min" and "entry_value_max" in optimisation_indicator:
            indicator["entry_value"] = int(round(random.uniform(config["indicators"][i]["entry_value_min"], config["indicators"][i]["entry_value_max"])))
        if "period_1_min" and "period_1_max" in optimisation_indicator:
            indicator["period_1"] = int(round(random.uniform(config["indicators"][i]["period_1_min"], config["indicators"][i]["period_1_max"])))
        if "period_2_min" and "period_2_max" in optimisation_indicator:
            indicator["period_2"] = int(round(random.uniform(config["indicators"][i]["period_2_min"], config["indicators"][i]["period_2_max"])))
        if "period_3_min" and "period_3_max" in optimisation_indicator:
            indicator["period_3"] = int(round(random.uniform(config["indicators"][i]["period_3_min"], config["indicators"][i]["period_3_max"])))
        if "gradient_threshold_min" and "gradient_threshold_max" in optimisation_indicator:
            indicator["gradient_threshold"] = random.uniform(config["indicators"][i]["gradient_threshold_min"], config["indicators"][i]["gradient_threshold_max"])
        if "use_gradient" in optimisation_indicator:
            indicator["use_gradient"] = config["indicators"][i]["use_gradient"]
        indicators.append(indicator)
    strategy_parameters["indicators"] = indicators if "indicators" in config else []
    end_trading_time = random.sample(list(custom_signals[0].keys()),1)[0] if custom_signals != [] else None
    strategy_parameters["signals"] = config["signals"] if "signals" in config else custom_signals[0][end_trading_time] if len(custom_signals) > 0 else custom_signals
    print("Random Strategy Parameters:", strategy_parameters) if print_output else None
    if len(custom_signals) > 1:
        all_strategy_parameters = []
        for i in range(len(custom_signals)):
            temp_strategy_parameters = strategy_parameters.copy()
            temp_strategy_parameters["signals"] = custom_signals[i][end_trading_time]
            all_strategy_parameters.append(temp_strategy_parameters)
        strategy_parameters = all_strategy_parameters
    if "signals" in config and len(config["signals"]) > 1:
        selected_signals = random.sample(list(config["signals"][0].keys()),1)[0]
        all_strategy_parameters = []
        for i in range(len(config["signals"])):
            temp_strategy_parameters = strategy_parameters.copy()
            temp_strategy_parameters["signals"] = config["signals"][i][selected_signals]
            all_strategy_parameters.append(temp_strategy_parameters)
        strategy_parameters = [all_strategy_parameters]       
    return strategy_parameters, end_trading_time.split("-")[-1] if end_trading_time is not None else None

def get_optimisation_parameters(custom_signals = []):
    with open('config.json') as json_file:
        config = json.load(json_file)    
    optimisation_parameters = config.copy()
    if "indicator_as_signal" in config:
        optimisation_parameters["indicator_as_signal"] = config["indicator_as_signal"]
    optimisation_parameters["indicators"] = config["indicators"] if "indicators" in config else []
    optimisation_parameters["signals"] = config["signals"] if "signals" in config else custom_signals      
    return optimisation_parameters

def get_model(r2_client, model_name = None):
    return next((model for model in r2_client.get_models() if model["name"] == model_name), None)

def get_models(r2_client):
    models = r2_client.get_models()
    selected_models = []
    for config_model in config["model"]:
        if "CDL" in config_model:
            selected_models.append(config_model)
        else:
            selected_models.append(next((model for model in models if model["name"] == config_model), None))
        print("\nLoaded", config_model, "Model")
    return selected_models

def request_helper(backtesting_parameters, other_parameters):
    optimisation_call_json = {**backtesting_parameters, **other_parameters}
    request_string = "?"
    for key, val in optimisation_call_json.items():
        request_string = request_string + f"{key}={val}&"
    return request_string


def get_charts_start_end(r2_client, chart_ids):
    chart_bars = [r2_client.get_chart(chart_id)["bars"] for chart_id in chart_ids]
    starts = [datetime.datetime.strptime(bars[0]["datetime"], "%Y-%m-%dT%H:%M:%S") for bars in chart_bars]
    ends = [datetime.datetime.strptime(bars[-1]["datetime"], "%Y-%m-%dT%H:%M:%S") for bars in chart_bars]
    return max([starts[i] for i in range(len(starts))]), min(
        [ends[i] for i in range(len(ends))]
    )

def post_custom_chart(r2_client, csv_file, interval, chart_name, stock_id, common=False):
    bars = pd.read_csv(csv_file)
    bars["datetime"] = pd.to_datetime(bars["datetime"], format='%d/%m/%Y %H:%M').dt.strftime('%Y-%m-%dT%H:%M:%S') if '/' in bars["datetime"][0] else bars["datetime"]
    chart = {"name": chart_name, "stock_id": stock_id, "common": common, "interval": int(interval), "bars":bars.to_dict("records")}
    for local_chart in r2_client.get_local_charts():
        if local_chart["name"] == chart["name"] and local_chart["interval"] == chart["interval"]:
            print("Chart with name", chart_name, "and interval", interval, "already exists.")
            return
    return r2_client.post_chart(chart)

def format_custom_bars(files = None):
    for file in files:
        df = pd.read_csv(file)
        df.rename(columns={'date.1':'datetime'}, inplace=True) 
        df2 = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df2.to_csv('custom_bars/'+file, index=False)
        
def save_results(all_cross_validation_windows, intervals, interval_dfs, csv_prefix = ""):
    final_dfs = []
    for i in range(len(interval_dfs)):
        interval_df = pd.read_csv("results/"+str(intervals[i])+"_full_results.csv", index_col='ticker')
        final_df, trades_df = interval_df.copy(), interval_df.copy()
        columns, sharpe_columns, trade_columns = [], [], []
        new_columns = {}
        for x in range(all_cross_validation_windows[i]):
            best_combined = str(x) + "_combined_validation_sharpe_ratio"
            best_column = str(x)+ "_validation_sharpe_ratio"
            best_trades = str(x)+ "_validation_trades"
            trade_columns.extend([best_trades])
            sharpe_columns.extend([best_combined])
            columns.extend([best_column])
            new_columns[best_column] = str(interval_df[str(x)+"_end"][0])+" - "+str(interval_df[str(x)+"_val_end"][0])
            new_columns[best_trades] = str(interval_df[str(x)+"_end"][0])+" - "+str(interval_df[str(x)+"_val_end"][0])
        final_df = final_df[columns].rename(columns=new_columns)
        final_df = final_df.rename(columns=new_columns)
        trades_df = trades_df[trade_columns].rename(columns=new_columns)
        trades_df = trades_df.rename(columns=new_columns)
        sharpe_df = pd.DataFrame(interval_df[sharpe_columns])
        sharpe_df = pd.DataFrame(np.reshape(sharpe_df.iloc[-1].values,(1,len(sharpe_df.iloc[-1].values))), columns=final_df.columns, index=['Combined Sharpe'])
        final_df = final_df.append(sharpe_df)
        trades_df.loc['Total Trades'] = trades_df.sum()
        final_df = final_df.append(trades_df.loc["Total Trades"])
        final_df.to_csv("results/"+csv_prefix+str(intervals[i])+'_results.csv')
        final_dfs.append(final_df)
    return final_dfs

def get_interval_start_end(interval, interval_df, custom_start_map):
    interval_df["start_datetime"], interval_df["end_datetime"] = pd.to_datetime(interval_df["start_datetime"]), pd.to_datetime(interval_df["end_datetime"])
    start = min(interval_df["start_datetime"])
    custom_start = custom_start_map[interval]
    end = max(interval_df["end_datetime"])
    if custom_start is not None:
        if start < custom_start < end:
            start = custom_start
        else:
            raise ValueError(f"Start date must be later than {start} and less than {end}")
    return start, end

def get_cross_validation_windows(start, end, interval, interval_days, validation_ratio):
    cross_validation_window_length = datetime.timedelta(days=interval_days)
    optimisation_length = (1-validation_ratio)*cross_validation_window_length
    validation_length = validation_ratio*cross_validation_window_length
    cross_validation_windows = int(((end-start)-cross_validation_window_length)/validation_length)
    print("\nInterval", interval, "from", start, "to", end, "with cross validation length", cross_validation_window_length, "gives", cross_validation_windows, "cross validation windows")
    return optimisation_length, validation_length, cross_validation_windows, cross_validation_window_length

def get_optimisation_ids(intervals, interval_dfs, all_cross_validation_windows, task_ids):
    counter = 0
    for x in range(len(intervals)):
        interval = intervals[x]
        interval_df = interval_dfs[x]
        for i in range(all_cross_validation_windows[x]):
            window_start = interval_df[str(i)+"_start"][0]
            interval_df[str(i)+"_optimisation_id"] = ""
            interval_df.loc[(interval_df["start_datetime"] <= window_start), str(i)+"_optimisation_id"] = task_ids[counter]
            counter+=1
        interval_df.to_csv("results/"+str(interval)+"_optimsation_ids.csv", date_format="%Y-%m-%dT%H:%M:%S")
    return interval_dfs

def save_custom_patterns(all_custom_signals, all_custom_patterns):
    with open("custom_signals.json", 'w') as f:
        json.dump(all_custom_signals, f, indent=4)  
    with open("custom_patterns.json", 'w') as f:
        json.dump(all_custom_patterns, f, indent=4) 
    print(f"Saved Custom Patterns")
        
def load_custom_patterns():
    with open("custom_signals.json") as json_file:
        all_custom_signals = json.load(json_file)
    with open("custom_patterns.json") as json_file:
        all_custom_patterns = json.load(json_file) 
    print(f"Loaded Custom Patterns\n")
    return all_custom_signals, all_custom_patterns

def format_custom_patterns(all_patterns, chart_csv, trade_direction):
    trade_direction_multiple = 1 if trade_direction == "long" else -1
    df = pd.read_csv(chart_csv).set_index("datetime")
    df.index = pd.DatetimeIndex(df.index).strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype({"close": float})
    start_time = config["start_trading_time"]
    end_time_min = config["end_trading_time_min"]
    end_time_max = config["end_trading_time_max"]
    start_time_dt = datetime.datetime.strptime(start_time, '%H:%M').time()
    end_time_max_dt = datetime.datetime.strptime(end_time_max, '%H:%M').time() 
    end_time_min_dt = datetime.datetime.strptime(end_time_min, '%H:%M').time()    
    all_patterns = [pattern for sublist in all_patterns for pattern in sublist]
    custom_signals = {pattern["shapes"][0]["points"][1]["datetime"]: {"confidence": pattern["confidence"], "price": pattern["shapes"][0]["points"][1]["price"], "height": abs(pattern["shapes"][0]["points"][1]["price"]-pattern["shapes"][0]["points"][0]["price"])} for pattern in all_patterns}
    dateTimeA = datetime.datetime.combine(datetime.date.today(), end_time_max_dt)
    dateTimeB = datetime.datetime.combine(datetime.date.today(), start_time_dt)
    dateTimeC = datetime.datetime.combine(datetime.date.today(), end_time_min_dt)
    dateTimeDifference = (dateTimeA - dateTimeB).total_seconds() / 60
    dateTimeDifference2 = (dateTimeC - dateTimeB).total_seconds() / 60    
    interval_mins = dateTimeDifference if end_time_min == end_time_max else 15
    interval = datetime.timedelta(minutes=interval_mins)
    all_custom_signals = {}
    for i in range(int(dateTimeDifference2/interval_mins)-1, int(dateTimeDifference/interval_mins)):
        signals = []
        end = (datetime.datetime.combine(datetime.date.today(), start_time_dt) + ((i+1)*interval)).time().strftime('%Y-%m-%dT%H:%M:%S')[11:16]
        for key, value in custom_signals.items():
            if key[11:16] <= end:
                signals.append({"datetime": key, "signal": value["confidence"]*trade_direction_multiple, "price": df.loc[key]["close"], "height": value["height"]})
        all_custom_signals[start_time+'-'+end] = signals
    return all_custom_signals, all_patterns

def get_custom_patterns(bars, custom_patterns, strategy_parameters):
    first_bar = bars['datetime'].values[0]
    last_bar = bars['datetime'].values[-1]
    patterns = []
    signals = strategy_parameters["signals"]
    for signal in signals:
        for pattern in custom_patterns:
            for shape in pattern["shapes"]:
                counter = 0
                if shape["shape_type"] == "trend_line" and counter == 0:
                    counter+=1
                    if shape["points"][1]["datetime"] == signal["datetime"]:
                        counter = 0
                        for shape in pattern["shapes"]:
                            if shape["shape_type"] == "trend_line" and counter == 0:
                                start_index1, start_price1, end_index1, end_price1 = get_pattern_info(shape)
                                counter += 1
                            elif shape["shape_type"] == "trend_line" and counter == 1:
                                start_index2, start_price2, end_index2, end_price2 = get_pattern_info(shape)
                                counter += 1  
                        if first_bar <= end_index1 <= last_bar:
                            p = {"end": end_index1, "start_price_1": start_price1,"end_price_1": end_price1, "start_price_2": start_price2, "end_price_2": end_price2, "confidence": pattern["confidence"]}
                            p["start_index"] = int(bars.index[bars['datetime'] == nearest_end(bars['datetime'], start_index1)].tolist()[0])
                            p["end_index"] = int(bars.index[bars['datetime'] == nearest_end(bars['datetime'], end_index1)].tolist()[0])
                            patterns.append(p)
    return patterns
  
def nearest_end(s, ts):
    i = bisect_left(s, ts)
    return min(s[max(0, i - 1) : i + 1], key=lambda t: t >= ts)
    
def get_pattern_info(shape):
    start_index = shape["points"][0]["datetime"]
    start_price = shape["points"][0]["price"]
    end_index = shape["points"][1]["datetime"]
    end_price = shape["points"][1]["price"]
    return start_index, start_price, end_index, end_price

def upload_chart_collection(r2_client, collection_name = None, csv_directory = None, replace = False, common=False):
    all_collections = [collection["name"] for collection in r2_client.get_collections()]
    if collection_name in all_collections:
        if not replace:
            print(f"Collection {collection_name} already exists. If you wish to replace it set replace = True")
            return
        else:
            print(f"Collection {collection_name} already exists. Replacing collection...")
            all_chart_names, all_intervals, chart_collection_id = r2_client.get_all_chart_info(collection = collection_name, print_output = False)
            print(r2_client._session.delete(r2_client._api_url + f"/chart-collections/{chart_collection_id}").json)
    else:
        print(f"Uploading chart collection {collection_name}...")
    chart_collection = {"name": collection_name,"common": common,"charts": []}
    for csv_file in os.listdir(csv_directory):
        csv_name = csv_file.split("_")
        chart_collection["charts"].append({"name": csv_name[0],"stock_id": csv_name[0],"interval": int(csv_name[-1].split(".")[0]),"bars": [pd.read_csv(csv_directory+"/"+csv_file).sort_values("datetime").to_dict("records")[0]]})
    r2_client._session.post(r2_client._api_url + "/chart-collections", json=chart_collection).json()
    all_chart_names, all_intervals, chart_collection_id = r2_client.get_all_chart_info(collection = collection_name, print_output = True)
    for csv_file in os.listdir(csv_directory):
        csv_name = csv_file.split("_")
        selected_charts, chart_ids = get_charts(r2_client, chart_collection_id, [csv_name[0]], int(csv_name[-1].split(".")[0]))
        print("DONE", csv_file, r2_client.update_chart(chart_ids[0], {"bars":pd.read_csv(csv_directory+"/"+csv_file)[1:].sort_values("datetime").to_dict("records")}))
    print(f"Finished uploading chart collection {collection_name}")