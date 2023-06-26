from flask import Flask, request, send_file, make_response
from flask_cors import CORS
import pyarrow as pa
import zlib
import json
import os
import datetime
import csv
import pandas as pd

app = Flask(__name__)
CORS(app)

uploadedFileName = None
uploadedFileType = None
df = None

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"msg": "file is not found"}, 500

    myFile = request.files["file"]

    global uploadedFileName, uploadedFileType, df

    uploadedFileName = myFile.filename
    uploadedFileType = myFile.content_type

    if uploadedFileType == "application/json":
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_json(myFile)
        
    elif uploadedFileType == "text/csv":
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_csv(myFile)

    elif uploadedFileType == "application/octet-stream" and myFile.filename.endswith(".parquet"):
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_parquet(myFile)
        df["datetime"] = df["datetime"].astype(str)
        


    # Slice the first 10 rows and save it as '10_rows.json'
    slicedData = df.head(10)
    slicedDataFilePath = "./public/10_rows.json"
    slicedData.to_json(slicedDataFilePath, orient="records")

    return {
        "file": myFile.filename,
        "path": f"/{myFile.filename}",
        "ty": myFile.content_type
    }

@app.route("/")
def index():
    if not uploadedFileName:
        return {"msg": "No file has been uploaded"}, 400

    filePath = "./public/10_rows.json"  # Change this to dynamic file path if needed
    headers = {}

    # Set the appropriate response headers for the compressed content
    headers["Content-Type"] = "application/json"

    response = make_response(send_file(filePath))
    # print('len(response):', len(response))
    response.headers = headers

    return response

@app.route("/sortData", methods=["POST"])
def sort_data():
    global df
    print('Streamed Data Length in Sort:', len(df))

    data = request.json

    print('Received parameters:', data)
    xAxisParam = data["xAxisParam"]
    yAxisParams = data["yAxisParams"]
    type = data.get("type")  # Use get method to retrieve the value with a default None if the key doesn't exist
    interval = data.get("interval")  # Use get method to retrieve the value with a default None if the key doesn't exist

    if not yAxisParams or len(yAxisParams) == 0:
        return "", 200

    sortedData = {}

    if type == "pie":
        groupedData = df.groupby(yAxisParams).size().reset_index(name='count')
        groupedData = groupedData.sort_values('count', ascending=False)
        print('groupedData:', groupedData)
        sortedData = {
            "xAxisData": [''.join(map(str, group)) for group in groupedData[yAxisParams].values],
            "yAxisData": groupedData['count'].values.tolist()
        }
        print('sorted data:', sortedData)
    else:
        print("type:", type)
        groupedData = {}
        for _, entry in df.iterrows():
            groupKey = entry[xAxisParam]

            if xAxisParam == "datetime" and interval == "daily":
                # Convert datetime string to datetime object
                groupKey = datetime.datetime.strptime(groupKey, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

            if xAxisParam == "datetime" and interval == "monthly":
                # Convert datetime string to datetime object
                groupKey = datetime.datetime.strptime(groupKey, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m")

            if xAxisParam == "datetime" and interval == "yearly":
                # Convert datetime string to datetime object
                groupKey = datetime.datetime.strptime(groupKey, "%Y-%m-%d %H:%M:%S").strftime("%Y")
                

            if groupKey not in groupedData:
                groupedData[groupKey] = {yAxisParam: 0 for yAxisParam in yAxisParams}

            for yAxisParam in yAxisParams:
                groupedData[groupKey][yAxisParam] += int(entry[yAxisParam])

        sortedData = {
            "xAxisData": sorted(groupedData.keys()),  # Sort the keys
            "yAxisData": [[groupedData[key][yAxisParam] for yAxisParam in yAxisParams] for key in sorted(groupedData.keys())]
        }
    
    return sortedData, 200

if __name__ == "__main__":
    app.run(port=5000)