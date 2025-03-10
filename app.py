import json
from flask import Flask, Response
from model import download_data
import os
from model import forecast_price

API_PORT = int(os.environ.get('API_PORT', 5000))

app = Flask(__name__)

def update_data():
    """Download price data, format data and train model."""
    tokens = ["ETH", "BNB", "ARB"]
    for token in tokens:
        download_data(token)
def get_token_inference(token):
    return forecast_price.get(token, 0)

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token not in ["ETH", "BNB", "ARB"]:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_token_inference(token)
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=API_PORT)
