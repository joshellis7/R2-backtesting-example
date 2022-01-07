from matplotlib.patches import Rectangle
import numpy as np
import mplfinance as mpf
import pandas as pd
import talib
import matplotlib.pyplot as plt

def format_bars(bars):
    formatted_bars = pd.DataFrame(bars)
    formatted_bars["datetime"] = pd.to_datetime(formatted_bars["datetime"], dayfirst=True)
    return formatted_bars


def get_index(dt, data):
    return data.index[data["datetime"] == pd.to_datetime(dt)][0]


def get_dates(data, start, end):
    if data["datetime"][0] <= start <= data["datetime"][len(data) - 1]:
        start = min(data["datetime"], key=lambda x: abs(x - start))
    else:
        raise ValueError("Start date not in chart")
    if data["datetime"][0] <= end <= data["datetime"][len(data) - 1]:
        end = min(data["datetime"], key=lambda x: abs(x - end))
    else:
        raise ValueError("End date not in chart")
    dates = pd.DatetimeIndex(data[get_index(start, data) : get_index(end, data)]["datetime"])
    return dates

def plot_macd(i, indicator, data, entry_level, colours):
    data["overbought"] = 0 + entry_level
    data["oversold"] = 0 - entry_level
    data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(data["close"], fastperiod=indicator["period_1"], slowperiod=indicator["period_2"], signalperiod=indicator["period_3"])
    data["macd"] = 100 * (data["macd"]) / data["close"]
    macd_plot = mpf.make_addplot(data["macd"], panel=1+i, color=colours[i], ylabel="MACD", secondary_y=False)
    overbought = mpf.make_addplot(data["overbought"], panel=1+i, color='white', secondary_y=False)
    oversold = mpf.make_addplot(data["oversold"], panel=1+i, color='white', secondary_y=False)    
    return [macd_plot, overbought, oversold]

def plot_rsi(i, indicator, data, entry_level, colours):
    data["overbought"] = 50 + entry_level
    data["oversold"] = 50 - entry_level
    data["rsi"] = talib.RSI(data["close"], timeperiod=indicator["period_1"])        
    rsi_plot = mpf.make_addplot(data["rsi"], panel=1+i, color=colours[i], ylabel="RSI", secondary_y=False)
    overbought = mpf.make_addplot(data["overbought"], panel=1+i, color='white', secondary_y=False)
    oversold = mpf.make_addplot(data["oversold"], panel=1+i, color='white', secondary_y=False)
    return [rsi_plot, overbought, oversold]

def plot_stochstic(i, indicator, data, entry_level, colours):
    data["overbought"] = 50 + entry_level
    data["oversold"] = 50 - entry_level        
    data["slowk"], data["slowd"] = talib.STOCH(data["high"],data["low"],data["close"],fastk_period=indicator["period_1"],slowk_period=indicator["period_2"],slowk_matype=0,slowd_period=indicator["period_3"],slowd_matype=0)       
    stochastic_plot1 = mpf.make_addplot(data["slowk"], panel=1+i, color=colours[i], ylabel="Stochastic", secondary_y=False)        
    stochastic_plot2 = mpf.make_addplot(data["slowd"], panel=1+i, color='b', secondary_y=False)
    overbought = mpf.make_addplot(data["overbought"], panel=1+i, color='white', secondary_y=False)
    oversold = mpf.make_addplot(data["oversold"], panel=1+i, color='white', secondary_y=False)
    return [stochastic_plot1, stochastic_plot2, overbought, oversold]

def plot_adx(i, indicator, data, entry_level, colours):
    data["strong_trend"] = entry_level
    data["adx"] = talib.ADX(data["high"],data["low"],data["close"],timeperiod=indicator["period_1"])       
    adx_plot = mpf.make_addplot(data["adx"], panel=1+i, color=colours[i], ylabel="ADX")        
    strong_trend = mpf.make_addplot(data["strong_trend"], panel=1+i, color='white')
    return [adx_plot, strong_trend]

def plot_equity(i, balance, equity, data):
    if len(balance)!=len(data):
        balance = np.pad(balance, (0, len(data)-len(balance)), 'constant', constant_values = (1,balance[-1]))
        equity = np.pad(equity, (0, len(data)-len(equity)), 'constant', constant_values = (1,equity[-1]))
    elif len(balance) == 0:
        balance, equity = np.ones(len(data)), np.ones(len(data))
    balance = mpf.make_addplot(balance, panel=2+i, color='limegreen', ylabel="Equity/Balance", secondary_y=False)
    equity = mpf.make_addplot(equity, panel=2+i, color='deepskyblue', secondary_y=False) 
    return [balance, equity]

def plot_patterns(data, patterns, start, end, backtesting_parameters, plotting_parameters, chart, models, strategy_parameters, balance, equity):
    dates = data["datetime"].to_list()
    data["datetime"] = pd.DatetimeIndex(data["datetime"])
    df = data.copy().set_index("datetime")
    colours = ['fuchsia','orange', 'yellow']
    plots = []
    i = -1
    for i in range(len(strategy_parameters["indicators"])):
        indicator = strategy_parameters["indicators"][i]
        entry_level = indicator["entry_value"]
        if indicator["name"] == 'macd':
            plots = plots + plot_macd(i, indicator, data, entry_level, colours)
        elif indicator["name"] == 'rsi':
            plots = plots + plot_rsi(i, indicator, data, entry_level, colours)        
        elif indicator["name"] == 'stochastic':
            plots = plots + plot_stochstic(i, indicator, data, entry_level, colours)        
        else:
            plots = plots + plot_adx(i, indicator, data, entry_level, colours) 
    ymin = min(data["low"])
    ymax = max(data["high"])
    height = ymax - ymin
    mc = mpf.make_marketcolors(up="limegreen", down="red", inherit=True)
    s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc)
    fig, axs = mpf.plot(df, type="candle", style=s, returnfig=True,addplot=plots + plot_equity(i, balance, equity, data), warn_too_much_data=len(df) + 1)    
    if plotting_parameters["show_patterns"]:
        for pattern in patterns:
            if type(pattern) != int:
                axs[0], in_dates = plot_rectangle(axs[0], data, pattern, dates)
                axs[0] = (
                    plot_trendlines(axs[0], data, pattern)
                    if in_dates and plotting_parameters["show_trendlines"]
                    else axs[0]
                )
    axs[0].set_ylim(ymin=ymin - (0.1 * height), ymax=ymax + (0.1 * height))
    axs[0].set_title(chart["name"] + ", " + str(chart["interval"]) + " + ".join([m["name"] if "CDL" not in m else m for m in models]))
    return axs[0]


def plot_rectangle(ax, data, pattern, dates):
    in_dates = False
    end_date = pattern["end"]
    if end_date in dates:
        width = pattern["end_index"] - pattern["start_index"]
        points = [pattern["start_price_1"], pattern["start_price_2"], pattern["end_price_1"], pattern["end_price_2"]]
        height = max(points) - min(points)
        rect = Rectangle(
            (pattern["start_index"], min(points)),
            width,
            height,
            linestyle="dashed",
            linewidth=1.5,
            edgecolor="deepskyblue",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            (pattern["start_index"] + (0.5 * width)),
            (min(points) + (1.025 * height)),
            str(round(pattern["confidence"] * 100, 2)) + "%",
            fontsize=8,
            color="deepskyblue",
            horizontalalignment="center",
        )
        in_dates = True
    return ax, in_dates


def plot_trendlines(ax, data, pattern):
    ax.plot(
        [pattern["start_index"], pattern["end_index"]],
        [pattern["start_price_1"], pattern["end_price_1"]],
        color="white",
        linewidth=1.5,
    )
    ax.plot(
        [pattern["start_index"], pattern["end_index"]],
        [pattern["start_price_2"], pattern["end_price_2"]],
        color="white",
        linewidth=1.5,
    )    
    return ax


def plot_trades(
    ax, bars, trades, tradesls, tradetps, results, backtest_parameters, strategy_parameters, plotting_parameters, end_index
):
    trades = pd.DataFrame(trades)
    if plotting_parameters["show_trades"] and len(trades) > 0:
        open_trades = trades.loc[pd.to_datetime(trades.entry_time) <= backtest_parameters["end_date"], :]
        ax = plot_trade_markers(ax, bars, open_trades, backtest_parameters["end_date"])
        ax = plot_sl_tp_levels(ax, tradesls, tradetps, end_index)
        ax = (
            plot_stats(ax, results, backtest_parameters, strategy_parameters, backtest_parameters["end_date"])
            if plotting_parameters["show_stats"]
            else ax
        )
    return ax


def plot_trade_markers(ax, bars, trades, end_date):
    if "type" in trades:
        buys = trades.loc[trades.type == "BUY", :]
        sells = trades.loc[trades.type == "SELL", :]
        if len(buys) > 0:
            ax.plot(
                buys["entry_index"],
                buys["entry_price"],
                "^",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1,
                alpha=1.0,
                markerfacecolor="limegreen",
                zorder=150,
                label="Buy",
            )
            if 'exit_price' in trades:
                ax.plot(
                    buys["exit_index"],
                    buys["exit_price"],
                    "v",
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    alpha=1.0,
                    markerfacecolor="limegreen",
                    zorder=100,
                    label="Close Buy",
            )
            if 'candle_index' in trades:
                buy_candles = bars.iloc[buys["candle_index"].tolist()]
                ax.plot(
                    buys["candle_index"],
                    buy_candles["low"],
                    "o",
                    markersize=10,
                    markeredgecolor="limegreen",
                    markeredgewidth=1,
                    alpha=1.0,
                    markerfacecolor="none",
                    zorder=100,
                    label="Buy Signal",
                )                 
        if len(sells) > 0:
            ax.plot(
                sells["entry_index"],
                sells["entry_price"],
                "v",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1,
                alpha=1.0,
                markerfacecolor="red",
                zorder=150,
                label="Sell",
            )
            if 'exit_price' in trades:
                ax.plot(
                    sells["exit_index"],
                    sells["exit_price"],
                    "^",
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    alpha=1.0,
                    markerfacecolor="red",
                    zorder=100,
                    label="Close Sell",
                )
            if 'candle_index' in trades:
                sell_candles = bars.iloc[sells["candle_index"].tolist()]
                ax.plot(
                    sells["candle_index"],
                    sell_candles["high"],
                    "o",
                    markersize=10,
                    markeredgecolor="red",
                    markeredgewidth=1,
                    alpha=1.0,
                    markerfacecolor="none",
                    zorder=100,
                    label="Sell Signal",
                )                  
        if 'exit_price' in trades:
            closed_buys = buys
            closed_sells = sells
            ax.plot(
                [closed_buys["entry_index"], closed_buys["exit_index"]],
                [closed_buys["entry_price"], closed_buys["exit_price"]],
                "--",
                color="limegreen",
            )
            ax.plot(
                [closed_sells["entry_index"], closed_sells["exit_index"]],
                [closed_sells["entry_price"], closed_sells["exit_price"]],
                "--",
                color="red",
            )
            for i in range(len(trades)):
                x0, x1 = trades["entry_index"][i], trades["exit_index"][i]
                y0, y1 = trades["entry_price"][i], trades["exit_price"][i]
                if y0 != y1:
                    y = np.arange(y0, y1, (y1 - y0) / 1000) if y1 > y0 else np.arange(y1, y0, (y0 - y1) / 1000)
                    if trades["bps_inc_fees_and_trade_size"][i] > 0:
                        ax.fill_betweenx(y, x0, x1, color="limegreen", alpha=0.6, zorder=-10, edgecolor="white", linewidth=2)
                    else:
                        ax.fill_betweenx(y, x0, x1, color="red", alpha=0.6, zorder=-10, edgecolor="white", linewidth=2)
    return ax


def plot_sl_tp_levels(ax, tradesls, tradetps, end_index):
    for sl in tradesls:
        sl = pd.DataFrame(sl)
        if len(sl) > 1:
            ax.plot(sl[0], sl[1], color="red", linewidth=2)
    for tp in tradetps:
        tp = pd.DataFrame(tp)
        if len(tp) > 1:
            ax.plot(tp[0], tp[1], color="limegreen", linewidth=2)
    return ax

def plot_live_sl_tp(ax, trades):
    levels = []
    for trade in trades:
        stop_loss = trade["stop_loss"]
        take_profit = trade["take_profit"]
        ax.plot(trade["entry_index"], stop_loss, "_", color="red", linewidth=2)
        ax.plot(trade["entry_index"], take_profit, "_", color="limegreen", linewidth=2)
        levels.append(stop_loss)
        levels.append(take_profit)
    ymin, ymax = ax.get_ylim()
    if len(levels) > 0:
        ymin = min(levels) if min(levels) < ymin else ymin
        ymax = max(levels) if max(levels) > ymax else ymax
        height = ymax-ymin
        ax.set_ylim(ymin=ymin - (0.1 * height), ymax=ymax + (0.1 * height))
    return ax


def plot_stats(ax, results, backtest_parameters, strategy_parameters, done):
    text = "\u0332".join("Backtest Parameters ")
    for key, value in backtest_parameters.items():
        text = text + "\n " + key + ": " + str(value)
    text = text + "\n \n"
    text = text + "\u0332".join("Strategy Parameters ")
    for key, value in strategy_parameters.items():
        if key == "indicators":
            for i in range(len(strategy_parameters["indicators"])):
                for inidicator_key, indicator_value in strategy_parameters["indicators"][i].items():
                    value = round(indicator_value, 2) if type(indicator_value) != bool and type(indicator_value) != str else indicator_value
                    text = text + "\n " + "indicator_"+str(i)+"_"+inidicator_key + ": " + str(value)
        elif key == "signals":
            None        
        else:
            value = round(value, 2) if type(value) != bool and type(value) != str else value
            text = text + "\n " + key + ": " + str(value)
    text = text + "\n \n"
    if done:
        text = (
            text
            + "\u0332".join("Backtest Performance ")
            + "\n Total BPS: "
            + str(round(results["total_bps"], 2))
            + "\n Sharpe Ratio: "
            + str(round(results["sharpe_ratio"], 2))
            + "\n Max Drawdown: "
            + str(round(results["max_drawdown"], 2))            
            + "\n Number of Trades: "
            + str(round(results["trades"]))
            + "\n Win %: "
            + str(round(results["win_percentage"], 2))
            + "%"
            + "\n Total Profit (bps): "
            + str(round(results["total_profit"], 2))
            + "\n Total Loss (bps): "
            + str(round(results["total_loss"], 2))
            + "\n Average Duration: "
            + str(results["average_duration"])
        )
    ax.text(
        1.01,
        0.99,
        text,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="black", alpha=1),
        label="Performance",
        size=6,
    )
    return ax

def plot_live_trades(r2_client, live_trade_request, last_updated, end_date, chart, model, backtesting_parameters):
    data = format_bars(r2_client.get_chart(chart["id"])["bars"])
    trades = pd.DataFrame(live_trade_request)
    patterns = trades["pattern"].tolist() if "pattern" in trades else []
    plotting_parameters = {
        "show_patterns": True,
        "show_trendlines": True,
        "show_trades": True,
        "show_stats": True,
    }
    backtesting_parameters["end_date"] = end_date
    backtesting_parameters["start_date"] = pd.to_datetime(sorted(patterns,key=lambda x: x["shapes"][0]["points"][0]["datetime"])[0]["shapes"][0]["points"][0]["datetime"]) if len(patterns) > 0 else last_updated
    ax = plot_patterns(data, patterns, backtesting_parameters, plotting_parameters, chart, model)
    ax = plot_trade_markers(ax,trades, end_date)
    ax = plot_live_sl_tp(ax, live_trade_request)


def plot_parameter_visualisation(optimisation_task_result):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update(mpl.rcParamsDefault)
    optimised_strategy_parameters = optimisation_task_result["best_parameters"]
    param_key = list(optimised_strategy_parameters.keys())
    trials = optimisation_task_result["trials"]
    fig, axs = plt.subplots(8, 4, sharey=True, figsize=(5, 10))
    results = trials["results"]
    parameter_counter = 0
    for i in range(len(param_key)-2):
        params = [trials["parameters"][x][i] for x in range(len(trials["results"]))]
        ax = axs[int(i / 4)][i % 4]
        if param_key[i] == "mean_reversion_or_breakout":
            params = ["breakout" if param == 0 else "mean_reversion" for param in params]
        s = ax.scatter(params, results, c=results, vmin=min(results), vmax=max(results))
        ax.plot(
            optimised_strategy_parameters[param_key[i]],
            optimisation_task_result["best_loss"],
            "o",
            c="red",
            markeredgecolor="black",
        )
        ax.set_title(param_key[i], fontsize=4)
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.tick_params(axis="both", which="minor", labelsize=4)
        parameter_counter+=1
    if "indicators" in optimised_strategy_parameters:
        indicators = optimised_strategy_parameters["indicators"]
        for y in range(len(indicators)):
            indicator = indicators[y]
            indicator_keys = list(indicator.keys())          
            for z in range(len(indicator_keys)-1):
                params = [trials["parameters"][x][parameter_counter] for x in range(len(trials["results"]))]
                ax = axs[int(parameter_counter / 4)][parameter_counter % 4]
                s = ax.scatter(params, results, c=results, vmin=min(results), vmax=max(results))
                ax.plot(
                    optimised_strategy_parameters["indicators"][y][indicator_keys[z]],
                    optimisation_task_result["best_loss"],
                    "o",
                    c="red",
                    markeredgecolor="black",
                )
                ax.set_title("Indicator " + str(y) + "-" + indicator_keys[z], fontsize=4)
                ax.tick_params(axis="both", which="major", labelsize=6)
                ax.tick_params(axis="both", which="minor", labelsize=4)
                parameter_counter+=1
    plt.suptitle("Parameter Visualisation", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
    cbar = fig.colorbar(s, cax=cbar_ax, label="Average bps")
    cbar.ax.tick_params(labelsize=6)
    fig.subplots_adjust(wspace=0.1)


def plot_candlestick_chart(data):
    df = data.set_index("datetime")
    mc = mpf.make_marketcolors(up="limegreen", down="red", inherit=True)
    s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc)
    fig, axs = mpf.plot(df, type="candle", style=s, returnfig=True, warn_too_much_data=len(df) + 1)
    return axs[0]

def plot_combined_equity(combined_result):
    plt.figure()
    all_dates = combined_result["all_dates"]
    all_balances_dt = pd.DataFrame(combined_result["balance"])
    all_balances_dt["datetime"] = pd.to_datetime(all_dates, dayfirst=True)
    all_balances_dt = all_balances_dt.set_index("datetime")
    all_equities_dt = pd.DataFrame(combined_result["equity"])
    all_equities_dt["datetime"] = pd.to_datetime(all_dates, dayfirst=True)
    all_equities_dt = all_equities_dt.set_index("datetime")
    plt.plot(all_balances_dt, color = 'limegreen', label = "Balance")
    plt.plot(all_equities_dt, color = 'deepskyblue', label = "Equity") 
    plt.legend()
    