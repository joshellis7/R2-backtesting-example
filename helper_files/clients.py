import requests
from fastapi import HTTPException
from requests.exceptions import HTTPError
import math
import json
import talib

with open('config.json') as json_file:
    config = json.load(json_file)
    
class R2Client:
    def __init__(self,):
        self._request_session = None
        base_url = config["r2_url"]
        self._api_url = base_url + "/financial/v1"
        auth_url = base_url + "/oauth/token"
        credentials = {
            "grant_type": "password",
            "username": config["r2_username"],
            "password": config["r2_password"],
            "client_id": config["r2_client_id"],
        }
        r = requests.post(auth_url, data=credentials)
        if r.status_code != 200:
            print(f"Status code: {r.status_code}.")
            print(r.json())
            raise Exception("Authentication failed.")
        tokens = r.json()

        refresh_data = {
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": credentials["client_id"],
            "username": credentials["username"],
        }

        r = requests.post(auth_url, data=refresh_data)
        if r.status_code != 200:
            print(f"Status code: {r.status_code}.")
            print(r.json())
            raise Exception("Authentication failed.")

        self.access_token = r.json()["access_token"]
        self._headers = {"Authorization": f"Bearer {self.access_token}"}

    @property
    def _session(self):
        if self._request_session is None:
            self._request_session = requests.Session()
            self._session.headers.update(self._headers)

        return self._request_session

    @staticmethod
    def _parse_response(response: requests.Response) -> dict:
        response.raise_for_status()
        return response.json()

    def get_models(self):
        return self._parse_response(self._session.get(self._api_url + "/models"))["models"]
    
    def get_all_models(self, print_output = False):
        models = self._parse_response(self._session.get(self._api_url + "/models"))["models"]
        model_names = list(set([model["name"] for model in models]))
        candlesticks = talib.get_function_groups()["Pattern Recognition"]
        if print_output:
            print("\nALL MODELS")
            print(model_names)
            print("\n")
            print("\nALL CANDLESTICKS")
            print(candlesticks)
            print("\n")            
        return model_names, candlesticks
    
    def get_model(self, model_id):
        models = self._parse_response(self._session.get(self._api_url + f"/models/{model_id}"))
        return models
    
    def get_collections(self):
        collections = self._parse_response(self._session.get(self._api_url + "/chart-collections"))["chart_collections"]
        return collections
    
    def get_collection_id(self, name:str):
        collections = self._parse_response(self._session.get(self._api_url + "/chart-collections"))["chart_collections"]
        collection = next((collection for collection in collections if collection["name"] == name), None)
        return collection["id"] if collection is not None else None

    def get_charts(self, collection: str):
        charts = self._parse_response(self._session.get(self._api_url + f"/chart-collections/{collection}"))["charts"]
        return charts
    
    def get_all_chart_info(self, collection = "Bloomberg", print_output = False):
        chart_collection_id = self.get_collection_id(collection)
        all_charts = self.get_charts(chart_collection_id)
        all_chart_names = list(set([chart["name"] for chart in all_charts]))
        all_intervals = list(set([chart["interval"] for chart in all_charts]))
        if print_output:
            print("\nALL CHARTS")
            print(all_chart_names)
            print("\n\n\nALL INTERVALS")
            print(all_intervals)
            print("\n")
        return all_chart_names, sorted(all_intervals), chart_collection_id
    
    def get_local_charts(self):
        charts = self._parse_response(self._session.get(self._api_url + "/charts"))["charts"]
        return charts
    
    def get_chart(self, chart_id: str):
        try:
            response = self._parse_response(self._session.get(self._api_url + f"/charts/{chart_id}?start_datetime=1990-01-01T00:00:00&max_bars=20000"))
            chart = response
            while len(response["bars"]) == 20000:
                response = self._parse_response(self._session.get(self._api_url + f"/charts/{chart_id}?start_datetime={response['bars'][-1]['datetime']}&max_bars=20000"))
                chart["bars"].extend(response["bars"])
        except HTTPError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Chart with id: {chart_id} does not exist.")
            else:
                raise
        return chart
    
    def get_all_charts(self):
        collections = self.get_collections()
        charts = []
        for collection in collections:
            charts.extend(self.get_charts(collection["id"]))
        return charts + self.get_local_charts()

    
    def post_chart(self, chart: str):
        bars = chart["bars"]
        if len(bars) > 20000:
            upload_windows = math.ceil(len(bars)/20000)
            window_size = int(round(len(bars)/upload_windows))
            for i in range(upload_windows):
                upload_bars = bars[i*window_size:(i+1)*window_size]
                if i == 0:
                    chart["bars"] = upload_bars
                    chart_id = self._parse_response(self._session.post(self._api_url + "/charts", json=chart))["id"]
                else:
                    self._session.patch(self._api_url + f"/charts/{chart_id}", json={"bars": upload_bars})
            return chart_id
        else:
            return self._parse_response(self._session.post(self._api_url + "/charts", json=chart))["id"]

    def update_chart(self, old_chart_id: str, new_chart):
        bars = new_chart["bars"]
        if len(bars) >= 20000:
            upload_windows = math.ceil(len(bars)/20000)
            window_size = int(round(len(bars)/upload_windows))
            for i in range(upload_windows):
                upload_bars = bars[i*window_size:(i+1)*window_size]
                new_chart["bars"] = upload_bars
                if i == 0:
                    chart_id = self._parse_response(self._session.post(self._api_url + "/charts", json=new_chart))["id"]
                else:
                    self._session.patch(self._api_url + f"/charts/{old_chart_id}", json={"bars": upload_bars})
            return chart_id
        else:
            return self._session.patch(self._api_url + f"/charts/{old_chart_id}", json={"bars": new_chart["bars"]})
        
    def delete_chart(self, chart_id):
        return self._session.delete(self._api_url + f"/charts/{chart_id}")