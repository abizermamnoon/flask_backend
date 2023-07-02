from flask import Flask, request, send_file, make_response
from flask_cors import CORS
import pyarrow as pa
import zlib
import json
import os
import datetime
import csv
import pandas as pd
import time

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

    global uploadedFileName, uploadedFileType, df, grouped_monthly, grouped_yearly, grouped_daily

    uploadedFileName = myFile.filename
    uploadedFileType = myFile.content_type

    if uploadedFileType == "application/json":
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_json(myFile)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    elif uploadedFileType == "text/csv":
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_csv(myFile)
        df['datetime'] = pd.to_datetime(df['datetime'])
        if isinstance(df, pd.DataFrame):
            print("Pandas DataFrame has been created.")
        else:
            print("Error: Failed to create Pandas DataFrame from CSV.")

    elif uploadedFileType == "application/octet-stream" and myFile.filename.endswith(".parquet"):
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_parquet(myFile)
        df["datetime"] = df["datetime"].astype(str)
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Perform groupby operations immediately after uploading the file
    grouped_monthly = group_by_monthly()
    grouped_yearly = group_by_yearly()
    grouped_daily = group_by_daily()
        
    # Slice the first 10 rows and save it as '10_rows.json'
    slicedData = df.head(5)
    slicedDataFilePath = "./public/10_rows.json"
    slicedData.to_json(slicedDataFilePath, orient="records")

    return {
        "file": myFile.filename,
        "path": f"/{myFile.filename}",
        "ty": myFile.content_type
    }

def group_by_monthly():
    global df
    return df.groupby(df['datetime'].dt.strftime('%Y-%m'))

def group_by_yearly():
    global df
    return df.groupby(df['datetime'].dt.strftime('%Y'))

def group_by_daily():
    global df
    return df.groupby(df['datetime'].dt.strftime('%Y-%m-%d'))


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
    start_time = time.time()
    global df, grouped_monthly, grouped_daily, grouped_yearly
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
        groupedData = df.groupby(yAxisParams[0]).size().reset_index(name='count')
        groupedData = groupedData.sort_values('count', ascending=False)
        print('groupedData:', groupedData)
        sortedData = {
            "xAxisData": [''.join(map(str, group)) for group in groupedData[yAxisParams].values],
            "yAxisData": groupedData['count'].values.tolist()
        }
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
    else:
        print("type:", type)
        groupedData = {}
        if xAxisParam != "datetime":
            grouped = df.groupby(xAxisParam)
            for groupKey, group in grouped:
                first_values = group.head(1)
                for _, entry in first_values.iterrows():
                    groupedData[groupKey] = {yAxisParam: entry[yAxisParam] for yAxisParam in yAxisParams}
        else:
            if interval == "daily":
                # df['datetime'] = pd.to_datetime(df['datetime'])
                grouped = grouped_daily
            elif interval == "monthly":
                # df['datetime'] = pd.to_datetime(df['datetime'])
                grouped = grouped_monthly
            elif interval == "yearly":
                # df['datetime'] = pd.to_datetime(df['datetime'])
                grouped = grouped_yearly
            else:
                return {"msg": "Invalid interval"}, 400
            
            for groupKey, group in grouped:
                first_values = group.head(1)
                for _, entry in first_values.iterrows():
                    groupedData[groupKey] = {yAxisParam: entry[yAxisParam] for yAxisParam in yAxisParams}
            
        sortedData = {
            "xAxisData": sorted(groupedData.keys()),  # Sort the keys
            "yAxisData": [[groupedData[key][yAxisParam] for yAxisParam in yAxisParams] for key in sorted(groupedData.keys())]
        }
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        
    
    return sortedData, 200

if __name__ == "__main__":
    app.run(port=5000)