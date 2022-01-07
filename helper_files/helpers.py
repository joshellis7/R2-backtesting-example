import datetime
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random
import json
import numpy as np

with open('config.json') as json_file:
    config = json.load(json_file)

def get_model(r2_client, model_name):
    models = r2_client.get_models()
    print("Loaded", model_name, "Model")
    return next((model for model in models if model["name"] == model_name), None)

def get_charts(r2_client, collection_id, names, interval):
    charts = r2_client.get_charts(collection_id)
    selected_charts = []
    for chart in charts:
        if any(name == chart["name"] for name in names) and chart["interval"] == interval:
            selected_charts.append(chart)
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
    
def get_backtesting_parameters(start, end, validation_period_ratio = 0.3, validation = False, print_output = True):
    if start is not None and end is not None:
        validation_start = start + (1-validation_period_ratio) * (end - start)
    backtesting_parameters = {
        "start_date": None if start is None else validation_start if validation else start,
        "end_date": None if end is None else end if validation else validation_start,
        "close_open_trades": config["close_open_trades"],
        "trade_fee": config["trade_fee"],
        "financing_fee": config["financing_fee"],
        "open_trade_limit": config["open_trade_limit"],
    }
    print("Backtesting Parameters:", backtesting_parameters, "\n") if print_output else None
    return backtesting_parameters

def get_random_strategy_parameters(interval, all_intervals, print_output = True):

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

    indicators = []
    for i in range(len(config["indicators"])):
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
    strategy_parameters["signals"] = config["signals"] if "signals" in config else []
    print("Random Strategy Parameters:", strategy_parameters) if print_output else None
    return strategy_parameters

def get_optimisation_parameters():
    optimisation_parameters = config
    optimisation_parameters["indicators"] = config["indicators"] if "indicators" in config else []
    optimisation_parameters["signals"] = config["signals"] if "signals" in config else []        
    return optimisation_parameters

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

def post_custom_chart(r2_client, csv_file, interval, chart_name, stock_id):
    bars = pd.read_csv(csv_file)
    bars["datetime"] = pd.to_datetime(bars["datetime"], format='%d/%m/%Y %H:%M').dt.strftime('%Y-%m-%dT%H:%M:%S') if '/' in bars["datetime"][0] else bars["datetime"]
    chart = {"name": chart_name, "stock_id": stock_id, "common": True, "interval": int(interval), "bars":bars.to_dict("records")}
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