# R2-backtesting-example

## Using the R2 Backtesting API
### Access
- An API key will be provided and must be used in the **Authorization** header of all requests.

### R2 API
- The backtesting API uses charts and models from your R2 API account.
- Custom models can be added to the R2 API using the UI.
- Custom charts can be added to the R2 API.

### Backtesting
- The backtesting endpoint returns backtesting performance results for specified backtest and strategy parameters.
- Please provide a `chart_id` and comma-seperated list of `model_ids` for you chosen chart and models from the R2 API. A `start_date` and `end_date` must also be provided and strategy parameters can be customised. Please see the docs for more information.

### Optimisation
- The optimisation endpoint returns optimised strategy parameters between defined strategy parameter limits for the specified charts and model.
- Please provide a comma-seperated list `chart_ids` and `model_ids` for you chosen charts and model from the R2 API. A `start_date` and `end_date` must also be provided and strategy parameter limits can be customised. Please see the docs for more information.
  
## Examples
- TA-Lib is required to use the example script. If you have not used TA-Lib before you will need to install the required dependencies before installing the package.

    If you are using Anaconda you can use
        ```
        conda install -c conda-forge ta-lib
        ```
    otherwise you will need to download TA-Lib before attempting to install `requirements.txt`. Please see the instructions provided in the **Dependencies** section of https://pypi.org/project/TA-Lib/ for more information.

- Setup a virtual environment with **Python 3.8** and install the requirements.

    **Using Pip:**
    ```
    virtualenv -p python3.8 <env_name>
    source <env_name>/bin/activate
    pip install -r requirements.txt
    ```
    **Using Anaconda:**
    ```
    conda create python=3.8 --name <env_name> --file requirements.txt
    ```

- Add your `backtesting_api_key`, `r2_username`, `r2_password`, `r2_client_id` to **config.json** and customise strategy parameters. Example strategy parameters can be found in **example_configs**.

- Run **examples.py** or **examples.ipynb**. For interactive plots add ```%matplotlib widget``` to the top of appropraite cells in **Jupyter Lab**.

## Docs
---

For more information please see the API docs:

**Trial**: http://trial-r2-backtesting-api.eba-xmxppepj.eu-west-1.elasticbeanstalk.com/docs


---