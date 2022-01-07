import requests
from helper_files.plotting_functions import format_bars, plot_patterns, plot_trades, plot_parameter_visualisation, plot_live_trades, plot_combined_equity
from helper_files.helpers import request_helper, get_optimisation_parameters, get_backtesting_parameters, get_interval_start_end, get_cross_validation_windows, get_optimisation_ids
import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()
import time
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
import json

with open('config.json') as json_file:
    config = json.load(json_file)
backtesting_url, backtesting_api_key = config["backtesting_url"], config["backtesting_api_key"]


result_columns = ["_optimisation_bps","_optimisation_sharpe_ratio","_optimisation_max_drawdown","_validation_bps","_validation_sharpe_ratio","_validation_max_drawdown", "_validation_trades"]


async def async_get_pattern_id(session, chart_id, model_id):
    try:
        chart_info = {"chart_id": chart_id, "model_id": model_id, "patterns": []}
        url = backtesting_url + "/pattern_collections/"
        async with session.post(url, json = chart_info, headers={"Authorization": backtesting_api_key}) as response:
            r = await response.json()
        return r["id"]
    except:
        return 0 

async def asynch_backtesting(
    chart_ids,
    backtesting_parameters,
    strategy_parameters,
    selected_charts,
    model,
    r2_client,
    plot_charts=False,
    prefix="",
    combine_results = False,
    print_output = True,
    save_output = False
):
    connector = aiohttp.TCPConnector(limit=10)
    client = aiohttp.ClientSession(connector=connector)
    async with client as session:
        tasks = []
        for i in range(len(chart_ids)):
            task = asyncio.ensure_future(
                get_backtesting(
                    session,
                    chart_ids[i],
                    backtesting_parameters,
                    strategy_parameters,
                    selected_charts[i],
                    model,
                    r2_client,
                    plot_charts,
                )
            )
            tasks.append(task)

        all_results = await asyncio.gather(*tasks)
    if print_output:
        print(
            prefix+" Backtest Complete. Average Sharpe Ratio =",
            sum([result["result"]["sharpe_ratio"] for result in all_results]) / len(all_results),
        )    
    combined_result = get_combined_results(all_results, backtesting_parameters, prefix=prefix, plot_equity=plot_charts) if combine_results else None
    if save_output:
        for i in range(len(chart_ids)):
            with open(f"results/{prefix}_{selected_charts[i]['name']}_results.json", 'w') as f:
                json.dump(all_results[i]["result"], f, indent=4)   
            trades = all_results[i]["trades"]
            for trade in trades:
                if "pattern" in trade:
                    trade.pop("pattern")
            with open(f"results/{prefix}_{selected_charts[i]['name']}_trades.json", 'w') as f:
                json.dump(trades, f, indent=4)                    
    return all_results


async def get_backtesting(
    session,
    chart_id,
    backtesting_parameters,
    strategy_parameters,
    chart,
    models,
    r2_client,
    plot,
):
    model_ids_string = (",").join([m["id"] if "CDL" not in m else m for m in models]) if len(models) > 0 else []
    url = backtesting_url + f"/backtest/run/{chart_id}/{model_ids_string}" + request_helper(backtesting_parameters, strategy_parameters)
    async with session.get(url, json={"indicators":strategy_parameters["indicators"], "signals":strategy_parameters["signals"]},headers={"Authorization": backtesting_api_key}, timeout=1800) as response:
        r = await response.json()
        print(r) if "result" not in r else None
        if plot:
            plotting_parameters = {
                "show_patterns": True,
                "show_trendlines": True,
                "show_trades": True,
                "show_stats": True,
            }
            result, trades, trade_sls, trade_tps, balance, equity = (
                r["result"],
                r["trades"],
                r["trade_sls"],
                r["trade_tps"],
                r["balance"],
                r["equity"],                
            )
            bars = format_bars(r2_client.get_chart(chart["id"])["bars"])
            patterns = [trade["pattern"] if "pattern" in trade else trade["candle_index"] for trade in trades]
            ax = plot_patterns(bars, patterns, result["start_date"], result["end_date"], backtesting_parameters, plotting_parameters, chart, models, strategy_parameters, balance, equity)
            plot_trades(
                ax,
                bars,
                trades,
                trade_sls,
                trade_tps,
                result,
                backtesting_parameters,
                strategy_parameters,
                plotting_parameters,
                len(bars),
            )
        return r


def start_optimisation(chart_ids, models, backtesting_parameters, interval, all_intervals):
    model_ids_string = (",").join([m["id"] if "CDL" not in m else m for m in models]) if len(models) > 0 else []
    chart_ids_string = (",").join(chart_ids)
    optimisation_parameters = get_optimisation_parameters()
    r = requests.post(
        backtesting_url
        + f"/optimise/run/{chart_ids_string}/{model_ids_string}"
        + request_helper(backtesting_parameters, optimisation_parameters),
        headers={"Authorization": backtesting_api_key},
        json={"indicators":optimisation_parameters["indicators"], "signals":optimisation_parameters["signals"]}
    ).json()
    print(r) if "task_id" not in r else None
    print("Optimisation Started")
    return r["task_id"], r["status"]

def get_optimisation_result(task_id, parameter_visualisation=False):
    result = requests.get(backtesting_url + f"/optimise/run/result/{task_id}", headers={"Authorization": backtesting_api_key})
    if result.status_code == 200 and parameter_visualisation:
        plot_parameter_visualisation(result.json()["result"])
    return result

def get_completed_optimisation_result(task_id, parameter_visualisation=False, print_output = True, save_parameters = True):
    optimisation_task_result = check_optimisation_result(task_id)
    counter = 0
    while optimisation_task_result.status_code != 200:
        print(optimisation_task_result.json()) if counter % 10 == 0 else None
        optimisation_task_result = check_optimisation_result(
            task_id, parameter_visualisation=parameter_visualisation
        )
        time.sleep(1)
        counter+=1
    optimisation_task_result = optimisation_task_result.json()["result"]
    optimised_strategy_parameters = optimisation_task_result["best_parameters"]
    if print_output:
        print("Optimisation Complete. Optimised Result =", optimisation_task_result["best_loss"])
        print("Optimised Parameters:", optimised_strategy_parameters, '\n')
    optimised_strategy_parameters["signals"] = config["signals"] if "signals" in config else []
    if save_parameters:
        with open('results/optimised_parameters.json', 'w') as f:
            json.dump(optimised_strategy_parameters, f, indent=4)         
    return optimisation_task_result, optimised_strategy_parameters

def check_optimisation_result(task_id, parameter_visualisation=False):
    result = requests.get(backtesting_url + f"/optimise/run/result/{task_id}", headers={"Authorization": backtesting_api_key})
    if result.status_code == 200 and parameter_visualisation:
        plot_parameter_visualisation(result.json()["result"])
    return result


async def get_live_trades(session, r2_client, chart, model, optimised_strategy_parameters, last_updated, end_date, backtesting_parameters, plot):
    url = backtesting_url + f"/live_trade/{chart['id']}/{model['id']}/{last_updated}" + request_helper(optimised_strategy_parameters,{})
    async with session.get(url, headers={"Authorization": backtesting_api_key}) as response:
        r = await response.json()
        if plot:
            plot_live_trades(r2_client, r, last_updated, end_date, chart, model, backtesting_parameters)
    return r


async def asynch_get_live_trades(
    r2_client,
    charts,
    model,
    strategy_parameters,
    last_updated,
    end_date,
    backtesting_parameters,
    plot=False,
):
    connector = aiohttp.TCPConnector(limit=10)
    client = aiohttp.ClientSession(connector=connector)
    async with client as session:
        tasks = []
        for i in range(len(charts)):
            task = asyncio.ensure_future(
                get_live_trades(
                    session,
                    r2_client,
                    charts[i],
                    model,
                    strategy_parameters,
                    last_updated,
                    end_date,
                    backtesting_parameters,
                    plot
                    
                )
            )
            tasks.append(task)
        all_results = await asyncio.gather(*tasks)

    return all_results

async def async_start_optimisation(session, chart_ids, models, backtesting_parameters):
    model_ids_string = (",").join([m["id"] if "CDL" not in m else m for m in models])
    chart_ids_string = (",").join(chart_ids)
    optimisation_parameters = get_optimisation_parameters()
    url = backtesting_url + f"/optimise/run/{chart_ids_string}/{model_ids_string}" + request_helper(backtesting_parameters, optimisation_parameters)
    tries, complete = False, 0
    while tries < 60 and not complete:
        async with session.post(url, json=optimisation_parameters["indicators"], headers={"Authorization": backtesting_api_key}, timeout=1800) as response:
            try:
                r = await response.json()
                task_id = r["task_id"]
                complete = True
                print(r)
            except Exception as error:
                time.sleep(1)
                tries+=1
                if tries % 10 == 0:
                    print("Too many optimisations submitted in 1 minute.", "API status:", response.status, "API response:", r, "Trying again...")
    return task_id if complete else ""

def start_interval_optimisation(i, tasks, session, interval, model_id, start, optimisation_length, validation_length, interval_df, backtesting_parameters):
    window_start = (start + validation_length*i)
    window_end = window_start + optimisation_length
    window_val_end = window_end + validation_length
    interval_df[str(i)+"_start"], interval_df[str(i)+"_end"], interval_df[str(i)+"_val_end"] = window_start, window_end, window_val_end
    backtesting_parameters["start_date"], backtesting_parameters["end_date"] = window_start, window_end
    active_chart_ids = interval_df.loc[interval_df["start_datetime"] <= window_start, 'chart_id'].tolist()
    tasks.append(asyncio.ensure_future(async_start_optimisation(session,active_chart_ids, model_id, backtesting_parameters)))
    print("Started Optimisation for interval", interval, "cross validation window", i+1)
    return tasks

async def start_all_optimisations(model_id, intervals, interval_dfs, validation_ratio = 0.3, cross_validation_start_15 = datetime.datetime(2021,1,1), cross_validation_start_60 = datetime.datetime(2018,1,1), cross_validation_start_240 = None, cross_validation_start_1440 = None, cross_validation_days_15 = int(365/4), cross_validation_days_60 = 365, cross_validation_days_240 = 2*365, cross_validation_days_1440 = 5*365):
    print("\nStarting Optimisations")
    starttime = time.time()
    custom_start_map = {15: cross_validation_start_15, 60: cross_validation_start_60, 240: cross_validation_start_240, 1440: cross_validation_start_1440}
    cv_days_map = {15: cross_validation_days_15, 60: cross_validation_days_60, 240: cross_validation_days_240, 1440: cross_validation_days_1440}    
    backtesting_parameters = get_backtesting_parameters(None, None, print_output = False)
    connector = aiohttp.TCPConnector(limit=1)
    client = aiohttp.ClientSession(connector=connector)
    async with client as session:
        all_cross_validation_windows = []
        tasks = []
        for x in range(len(intervals)):
            start, end = get_interval_start_end(intervals[x], interval_dfs[x], custom_start_map)
            optimisation_length, validation_length, cross_validation_windows, cross_validation_window_length = get_cross_validation_windows(start, end, intervals[x], cv_days_map[intervals[x]], validation_ratio)
            all_cross_validation_windows.append(cross_validation_windows)
            for i in range(cross_validation_windows):
                tasks = start_interval_optimisation(i, tasks, session, intervals[x], model_id, start, optimisation_length, validation_length, interval_dfs[x], backtesting_parameters)
        task_ids = await asyncio.gather(*tasks)
    interval_dfs = get_optimisation_ids(intervals, interval_dfs, all_cross_validation_windows, task_ids)
    print("\nStarted All Optimisations, Took", str(round(time.time()-starttime)) + "s")
    return all_cross_validation_windows, interval_dfs

def get_cross_validation_results(models, all_cross_validation_windows, intervals, interval_dfs, r2_client, save_results = False):
    print("\nGetting Optimisation & Cross Validation Results")
    starttime = time.time()
    backtesting_parameters = get_backtesting_parameters(None, None, print_output = False)
    for z in range(len(interval_dfs)):
        interval, interval_df = intervals[z], interval_dfs[z]
        for i in range(all_cross_validation_windows[z]):
            optimisation_task_ids = list(filter(None, interval_df[str(i)+"_optimisation_id"].unique().tolist()))
            new_result_columns = [col for col in result_columns]
            for x in range(len(optimisation_task_ids)):
                optimisation_task_result, optimised_strategy_parameters = get_completed_optimisation_result(optimisation_task_ids[x], print_output = False)                
                group_chart_ids = interval_df.loc[interval_df[str(i)+"_optimisation_id"] == optimisation_task_ids[x], 'chart_id']
                chart_name = group_chart_ids.index[0]
                group_chart_ids = group_chart_ids.tolist()
                if optimised_strategy_parameters is not None:
                    backtesting_parameters["start_date"], backtesting_parameters["end_date"] = interval_df[str(i)+"_end"][0], interval_df[str(i)+"_val_end"][0]
                    validation_result = asyncio.run(
                        asynch_backtesting(
                            group_chart_ids,
                            backtesting_parameters,
                            optimised_strategy_parameters,
                            group_chart_ids,
                            models,
                            r2_client,
                            plot_charts=False,
                            prefix=str(interval) + " Validation: Interval " + str(i),
                            print_output=False,
                            combine_results=False
                        )
                    )
                    combined_result = get_combined_results(validation_result, backtesting_parameters, plot_equity=False, print_output=False)
                    if save_results:
                        with open('results/'+str(interval)+"_"+chart_name+"_"+str(i)+'_validation_result.json', 'w') as f:
                            json.dump(validation_result[0], f)          
                        with open('results/'+str(interval)+"_"+chart_name+"_"+str(i)+'_optimised_parameters.json', 'w') as f:
                            json.dump(optimised_strategy_parameters, f)                    
                else:
                    validation_result = [{} for i in range(len(group_chart_ids))]
                interval_df.loc[interval_df[str(i)+"_optimisation_id"] == optimisation_task_ids[x], [str(i)+result_column for result_column in new_result_columns[-4:]]] = [[result["result"]["total_bps"], result["result"]["sharpe_ratio"], result["result"]["max_drawdown"], len(result["trades"])] if "result" in result else [0,0,0,0] for result in validation_result]
                interval_df.loc[interval_df[str(i)+"_optimisation_id"] == optimisation_task_ids[x], [str(i)+"_combined_validation_sharpe_ratio"]] = combined_result["sharpe_ratio"]
            print("Got Results for Interval", interval,"Cross Validation Window", i)
            interval_df.to_csv("results/"+str(interval)+"_full_results.csv", date_format="%Y-%m-%dT%H:%M:%S")
    print("Got all Cross Validation Results, Took", str(round(time.time()-starttime)) + "s")
    return interval_dfs


async def get_patterns(
    session,
    chart_id,
    model,
    r2_client
):
    model_id = model["id"] if "id" in model else model
    url = backtesting_url + f"/pattern_collections/start_end/{chart_id}/{model_id}"
    async with session.get(url,headers={"Authorization": backtesting_api_key}, timeout=1800) as response:
        r = await response.json()
        try:
            return [r["start_date"], r["end_date"]]
        except:
            bars = r2_client.get_chart(chart_id)["bars"]
            return [bars[0]['datetime'],bars[-1]['datetime']]
        
async def get_start_end_dates(
    chart_ids,
    models,
    r2_client
):
    models = ["None"] if models == [] else models
    connector = aiohttp.TCPConnector(limit=10)
    client = aiohttp.ClientSession(connector=connector)
    async with client as session:
        tasks = []
        for i in range(len(chart_ids)):
            for x in range(len(models)):
                task = asyncio.ensure_future(get_patterns(session,chart_ids[i],models[x],r2_client))
                tasks.append(task)
        all_results = await asyncio.gather(*tasks)

    starts = [result[0] for result in all_results]
    ends = [result[-1] for result in all_results]
    return datetime.datetime.strptime(max([starts[i] for i in range(len(starts))]), "%Y-%m-%dT%H:%M:%S"),  datetime.datetime.strptime(min([ends[i] for i in range(len(ends))]), "%Y-%m-%dT%H:%M:%S")

def get_combined_results(backtesting_results, backtesting_parameters, prefix = "", plot_equity=False, print_output=True):
    result = requests.get(backtesting_url + "/backtest/combine"+ request_helper(backtesting_parameters, {}), json = backtesting_results, headers={"Authorization": backtesting_api_key}).json()
    print(result) if "sharpe_ratio" not in result else None
    plot_combined_equity(result) if plot_equity else None
    print(prefix+" Combined Backtest Complete. Sharpe Ratio =",result["sharpe_ratio"]) if print_output else None
    return result
    