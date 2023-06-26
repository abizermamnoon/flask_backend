from flask import Flask, request, send_file, make_response
from flask_cors import CORS
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
streamedData = []

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"msg": "file is not found"}, 500

    myFile = request.files["file"]

    global uploadedFileName, uploadedFileType, streamedData

    uploadedFileName = myFile.filename
    uploadedFileType = myFile.content_type


    myFile.save(f"./public/{myFile.filename}")

    filePath = f"./public/{myFile.filename}"

    if uploadedFileType == "application/json":
        print('Uploaded File Type:', uploadedFileType)
        with open(filePath, "r") as file:
            streamedData = json.load(file)
    elif uploadedFileType == "text/csv":
        print('Uploaded File Type:', uploadedFileType)
        with open(filePath, "r") as file:
            csv_reader = csv.DictReader(file)
            streamedData = [row for row in csv_reader]
    elif uploadedFileType == "application/octet-stream" and myFile.filename.endswith(".parquet"):
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_parquet(filePath)
        df["datetime"] = df["datetime"].astype(str)
        streamedData = df.to_dict(orient="records")

    # Slice the first 10 rows and save it as '10_rows.json'
    slicedData = streamedData[:10]
    slicedDataFilePath = "./public/10_rows.json"
    with open(slicedDataFilePath, "w") as slicedFile:
        json.dump(slicedData, slicedFile)

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
    
    if uploadedFileType == "application/json":
        headers["Content-Type"] = "application/json"
    elif uploadedFileType == "text/csv":
        headers["Content-Type"] = "text/csv"

    response = make_response(send_file(filePath))
    # print('len(response):', len(response))
    response.headers = headers

    return response

@app.route("/sortData", methods=["POST"])
def sort_data():
    global streamedData
    print('Streamed Data Length in Sort:', len(streamedData))

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
        print("type:", type)
        groupedData = {}
        for entry in streamedData:
            groupKey = tuple(entry[yAxisParam] for yAxisParam in yAxisParams)
            
            if groupKey not in groupedData:
                groupedData[groupKey] = {yAxisParam: 0 for yAxisParam in yAxisParams}

            for yAxisParam in yAxisParams:
                groupedData[groupKey][yAxisParam] += 1

        sortedData = {
            "xAxisData": ["".join(groupKey) for groupKey in sorted(groupedData.keys(), key=lambda k: groupedData[k][yAxisParams[0]], reverse=True)],
            "yAxisData": [[groupedData[key][yAxisParam] for yAxisParam in yAxisParams] for key in sorted(groupedData.keys(), key=lambda k: groupedData[k][yAxisParams[0]], reverse=True)]
        }
    else:
        groupedData = {}
        print("type:", type)
        for entry in streamedData:
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