import datetime
import asyncio
import nest_asyncio
nest_asyncio.apply()
from helper_files.r2_backtesting_calls import asynch_backtesting, start_optimisation, get_completed_optimisation_result, get_start_end_dates
from helper_files.clients import R2Client
from helper_files.helpers import get_models, get_charts, get_backtesting_parameters, get_random_strategy_parameters
r2_client = R2Client() #Authorisation is only valid for 1 hour. Please run this line to re-authorise

# In[1] Choose Charts, model and interval

# Get names of all charts, intervals and models/candlestick patterns to choose from
all_chart_names, all_intervals, chart_collection_id = r2_client.get_all_chart_info(print_output = True)
all_models, all_candlesticks = r2_client.get_all_models(print_output = True)

# Choose Charts + Interval
chart_names = ['CL1 Comdty', 'HO1 Comdty', 'NG1 Comdty']
interval = 15 # Choose from 15, 60, 240, 1440

# In[2] Get Model ID + Chart IDs from R2 API

models = get_models(r2_client)
selected_charts, chart_ids = get_charts(r2_client, chart_collection_id, chart_names, interval)

# In[3]
# Set Start and End Date for Backtesting
start, end = asyncio.run(get_start_end_dates(chart_ids, models, r2_client)) # set start and end dates as the start and end of the chart
#start, end = datetime.datetime(2021,2,1), datetime.datetime(2021,5,1) # set custom start and end dates

# Get Backtesting Parameters
backtesting_parameters = get_backtesting_parameters(start, end, validation_period_ratio = 0.3, validation = False, print_output = True)

# Get Random Strategy Parameters
random_strategy_parameters = get_random_strategy_parameters(interval, all_intervals, print_output = True)

# Run async Backtesting on Selected Charts and Model using Backtest and Random Strategy Parameters
backtesting_result = asyncio.run(
    asynch_backtesting(
        chart_ids,
        backtesting_parameters,
        random_strategy_parameters,
        selected_charts,
        models,
        r2_client,
        plot_charts=True,
        prefix="Random",
        combine_results = True,
        save_output = True
    )
)

# In[4] Start Optimisation and get Result
# Start Optimisation on Selected Charts and Model
optimisation_task_id, status = start_optimisation(chart_ids, models, backtesting_parameters, interval, all_intervals)

# Check if Optimisation has finished
optimisation_task_result, optimised_strategy_parameters = get_completed_optimisation_result(optimisation_task_id, parameter_visualisation=True, save_parameters=True)

# Run async Backtesting on Selected Charts and Model using Optimised Strategy Parameters on Optimsation Time Period
optimisation_result = asyncio.run(
    asynch_backtesting(
        chart_ids,
        backtesting_parameters,
        optimised_strategy_parameters,
        selected_charts,
        models,
        r2_client,
        plot_charts=True,
        prefix="Optimised",
        combine_results = True,
        save_output = True
    )
)

# In[5] Run async Backtesting with Optimised Parameters on Out of Sample Test Period.

validation_backtesting_parameters = get_backtesting_parameters(start, end, validation_period_ratio = 0.3, validation = True, print_output = True)
validation_result = asyncio.run(
    asynch_backtesting(
        chart_ids,
        validation_backtesting_parameters,
        optimised_strategy_parameters,
        selected_charts,
        models,
        r2_client,
        plot_charts=True,
        prefix="Validation",
        combine_results = True,
        save_output = True
    )
)
