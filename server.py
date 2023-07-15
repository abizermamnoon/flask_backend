from flask import Flask, request, send_file, make_response, Response, jsonify
from flask_cors import CORS
from io import StringIO
import pyarrow as pa
import zlib
import json
import os
import datetime
import csv
# import pandas as pd
import time
import numpy as np
import copy
import pandas as pd
app = Flask(__name__)
CORS(app)

uploadedFileName = None
uploadedFileType = None
df = None
filtered_df = None
state = {}
state = {
    "frame": None
}
frame_rep = {} 
response = {}
column = None
converted_columns = []
grouped_data = {}
data_types = {}
sortedData = {}
xAxisParam = None
yAxisParams = []
type = None
xAxisParam_1 = None
yAxisParams_1 = []
state_1 = {}
state_1 = {
    "frame": None
}
frame_rep_1 = {} 
headers = []
type_1 = None
pie_groupedData = None
groupedData = {}


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"msg": "file is not found"}, 500

    myFile = request.files["file"]

    global uploadedFileName, uploadedFileType, df, grouped_monthly, grouped_yearly, grouped_daily, state, frame_rep, response, grouped_data, data_types

    uploadedFileName = myFile.filename
    uploadedFileType = myFile.content_type

    if uploadedFileType == "application/json":
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_json(myFile)
        df = convert_datetime_columns(df, ['%Y-%m-%d %H:%M:%S', '%m-%d-%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d'])
        
    elif uploadedFileType == "text/csv":
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_csv(myFile)
        df = convert_datetime_columns(df, ['%Y-%m-%d %H:%M:%S', '%m-%d-%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d'])
        # df['datetime'] = pd.to_datetime(df['datetime'])
        if isinstance(df, pd.DataFrame):
            print("Pandas DataFrame has been created.")
        else:
            print("Error: Failed to create Pandas DataFrame from CSV.")
        
    elif uploadedFileType == "application/octet-stream" and myFile.filename.endswith(".parquet"):
        print('Uploaded File Type:', uploadedFileType)
        df = pd.read_parquet(myFile)
        df = convert_datetime_columns(df, ['%Y-%m-%d %H:%M:%S', '%m-%d-%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d'])

    return {
        "file": myFile.filename,
        "path": f"/{myFile.filename}",
        "ty": myFile.content_type
    }

def convert_datetime_columns(df, formats):
    datetime_columns = df.select_dtypes(include=[object]).columns
    global converted_columns
    # print('datetime_columns:', datetime_columns)
    for column in datetime_columns:
        for fmt in formats:
            try:
                df[column] = pd.to_datetime(df[column], format=fmt)
                converted_columns.append(column)
            except ValueError:
                pass
    print('datetime_columns:', converted_columns)
    return df

@app.route("/nullval", methods=["POST"])
def countnul():
    global df
    if df is not None:
        df_nul = df.isnull().sum()
        df_nul = df_nul.to_frame(name='count_nuls').reset_index()
        df_nul.columns = ['columns', 'count_nuls']
        df_nul = df_nul[df_nul['count_nuls'] > 0]
        print('df_nul:', df_nul)
        formattedData = format_isnul(df_nul)
        return jsonify(formattedData)
    else:
        return 'Dataframe not uploaded'

def format_isnul(df):
    state = {}
    state = {
        "frame": None
    }
    headers = ['columns', 'count_nuls']

    frame_rep = dict()
    frame_rep["columns"] = [{
        "Header": column,
        "accessor": column
    } for column in headers]
    frame_rep["data"] = []

    state["frame"] = df
    print('state:', state["frame"])
    for _, row in state["frame"].iterrows():
        formatted_row = {}
        for column, value in row.items():               
            formatted_row[column] = value
        frame_rep["data"].append(formatted_row)
    
    print('frame_rep:', frame_rep)

    return frame_rep

@app.route("/dropna", methods=["POST"])
def findna():
    data = request.json
    action = data['action']
    print('Requested Data:', data)

    if action == 'drop':
        dropna()
        return "Rows with NA values have been dropped"
    if action == 'next':
        nextna()
        return "cells with NA values have been replaced by rows below it"
    if action == 'prev':
        prevna()
        return "cells with NA values have been replaced by rows above it"
    if action == 'interp':
        interpna()
        return "cells with NA values have been replaced by linear interpolation"
    
def dropna():
    global df
    df = df.dropna()

def nextna():
    global df
    df = df.fillna(method ='bfill')

def prevna():
    global df
    df = df.fillna(method ='pad')

def interpna():
    global df
    df = df.interpolate(method ='linear', limit_direction ='forward')


@app.route("/loadTable", methods=["POST"])
def load():

    global df, grouped_monthly, grouped_yearly, grouped_daily, state, frame_rep, response, grouped_data, data_types

    start_time = time.time()

    state["frame"] = df.head(100)
    frame_rep = formatFrame()
    response = jsonify(frame_rep)

    # Perform groupby operations immediately after uploading the file for all columns
    
    for column in df.columns:
        grouped_data[column] = df.groupby(column)

    grouped_monthly = group_by_monthly()
    grouped_yearly = group_by_yearly()
    grouped_daily = group_by_daily()
        
    # Slice the first 10 rows and save it as '10_rows.json'
    slicedData = df.head(5)
    slicedDataFilePath = "./public/10_rows.json"
    slicedData.to_json(slicedDataFilePath, orient="records")

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    return "Data has been loaded"

def group_by_monthly():
    global df, converted_columns
    if len(converted_columns) > 0:
        grouped_monthly_data = {}
        for column in converted_columns:
            grouped_monthly_data[column] = df.groupby(df[column].dt.strftime('%Y-%m'))
        return grouped_monthly_data
    else:
        return None

def group_by_yearly():
    global df, converted_columns
    if len(converted_columns) > 0:
        grouped_yearly_data = {}
        for column in converted_columns:
            grouped_yearly_data[column] = df.groupby(df[column].dt.strftime('%Y'))
        return grouped_yearly_data
    else:
        return None

def group_by_daily():
    global df, converted_columns
    if len(converted_columns) > 0:
        grouped_daily_data = {}
        for column in converted_columns:
            grouped_daily_data[column] = df.groupby(df[column].dt.strftime('%Y-%m-%d'))
        return grouped_daily_data
    else:
        return None

def formatFrame():
    frame_rep = dict()
    frame_rep["columns"] = [{
        "Header": column,
        "accessor": column
    } for column in state["frame"].columns]
    frame_rep["data"] = []
    for _, row in state["frame"].iterrows():
        formatted_row = {}
        for column, value in row.items():
            if isinstance(value, pd.Timestamp):
                formatted_row[column] = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, bool):
                formatted_row[column] = str(value)
            else:
                formatted_row[column] = value
        frame_rep["data"].append(formatted_row)
    print("Frame formatted")
    return frame_rep

@app.route("/")
def index():
    
    if not uploadedFileName:
        return {"msg": "No file has been uploaded"}, 400

    filePath = "./public/10_rows.json"  # Change this to dynamic file path if needed
    headers = {}

    # Set the appropriate response headers for the compressed content
    headers["Content-Type"] = "application/json"
    
    response = make_response(send_file(filePath))
    response.headers = headers

    return response

@app.route("/coltype", methods=["POST"])
def findColType():
    global df
    if df is not None:
        dtypes_dict = df.dtypes.apply(lambda x: map_dtype_name(x)).to_dict()
        return jsonify(dtypes_dict)
    else:
        return 'Dataframe has not been uploaded'

def map_dtype_name(dtype):
    if np.issubdtype(dtype, np.integer):
        return "int"
    elif np.issubdtype(dtype, np.floating):
        return "int"
    elif np.issubdtype(dtype, np.object_):
        return "string"
    elif np.issubdtype(dtype, np.bool_):
        return "string"
    elif np.issubdtype(dtype, np.datetime64):
        return "datetime"
    else:
        return str(dtype)

@app.route("/sortData", methods=["POST"])
def sort_data():
    start_time = time.time()
    global df, grouped_monthly, grouped_daily, grouped_yearly, grouped_data, converted_columns, xAxisParam, yAxisParams, xAxisParam_1, yAxisParams_1, type_1, type
    if df is not None:
        print('Streamed Data Length in Sort:', len(df))

    data = request.json

    print('Received parameters:', data)
    xAxisParam = data["xAxisParam"]
    yAxisParams = data["yAxisParams"]
    type = data.get("type")   # Use get method to retrieve the value with a default None if the key doesn't exist
    interval = data.get("interval")  # Use get method to retrieve the value with a default None if the key doesn't exist

    if data["xAxisParam"] and len(data["xAxisParam"]) > 1:
        xAxisParam_1 = copy.copy(xAxisParam)
        print('X-Axis Parameter:', xAxisParam_1)
    if data["yAxisParams"] != []:
        yAxisParams_1 = copy.copy(yAxisParams)
        print('Y-Axis Parameter:', yAxisParams_1)
    if type:
        type_1 = copy.copy(type)
        print('Chart Type:', type_1)
    

    if not yAxisParams or len(yAxisParams) == 0:
        return "", 200
    
    global sortedData, pie_groupedData, groupedData

    if len(xAxisParam) == 0:
        pie_groupedData = df.groupby(yAxisParams[0]).size().reset_index(name='count')
        pie_groupedData = pie_groupedData.sort_values('count', ascending=False)
        print('pie_groupedData:', pie_groupedData)
        sortedData = {
            "xAxisData": [''.join(map(str, group)) for group in pie_groupedData[yAxisParams].values],
            "yAxisData": pie_groupedData['count'].values.tolist()
        }
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
    else:
        
        if xAxisParam not in converted_columns:
           groupedData = {}
           grouped = grouped_data[xAxisParam]
            
        else:
            groupedData = {}
            if interval == "daily":
                grouped = grouped_daily[xAxisParam]                
            elif interval == "monthly":
                grouped = grouped_monthly[xAxisParam]                
            elif interval == "yearly":
                grouped = grouped_yearly[xAxisParam]               
            else:
                return {"msg": "Invalid interval"}, 400
            
        for groupKey, group in grouped:
            first_values = group.head(1)
            for _, entry in first_values.iterrows():
                groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
        print('grouped data:', groupedData)
            
        sortedData = {
            "xAxisData": sorted(groupedData.keys()),  # Sort the keys
            "yAxisData": [[groupedData[key][yAxisParam] for yAxisParam in yAxisParams] for key in sorted(groupedData.keys())]
        }
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        
    
    return sortedData, 200


@app.route('/create', methods=["POST"])
def createFrame():
    global response
    
    print("Frame created")
    return response

@app.route('/get_groups', methods=["POST"])
def findGroups():
    global grouped_data, column, df
    data = request.json

    column = data["column"]

    column_data_type = df.dtypes[column]
    print('column data type:', column_data_type)

    if column in grouped_data and column_data_type in ['object', 'bool']:
        groups = list(grouped_data[column].groups.keys())
        groups = [str(group) for group in groups]
    print('groups:', groups)
        
    return jsonify(groups)
    
@app.route('/filter', methods=["POST"])
def findFilter():
    global df, column, groups
    data = request.json

    groups = data["group"]
    

    if groups == ['False']:
        groups = [False]
    elif groups == ['True']:
        groups = [True]

    print('groups:', groups)

    if groups:
        filtered_df = pd.DataFrame()  # Create an empty DataFrame for storing the filtered results
        for group in groups:
            temp_df = df[df[column] == group]  # Filter the DataFrame for each group
            filtered_df = filtered_df.append(temp_df)  # Append the filtered results to the main DataFrame
        print('First 5 rows of filtered dataframe:', filtered_df.head(5))
        print('Last 5 rows of filtered dataframe:', filtered_df.tail(5))
    else:
        filtered_df = df

    # Convert the datetime column to string format
    if 'datetime' in filtered_df.columns:
        filtered_df['datetime'] = filtered_df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Convert the filtered DataFrame to the response format
    response = {
        "columns": filtered_df.columns.tolist(),
        "data": format_dataframe(filtered_df.head(100))
    }

    return jsonify(response)

def format_dataframe(df):
    formatted_data = []
    for _, row in df.iterrows():
        formatted_row = {}
        for column, value in row.items():
            if isinstance(value, pd.Timestamp):
                formatted_row[column] = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, bool):
                formatted_row[column] = str(value)
            else:
                formatted_row[column] = value
        formatted_data.append(formatted_row)
    return formatted_data

@app.route('/calculation', methods=["POST"])

def equation():
    global df, filtered_df

    data = request.json
    eq = data["calculation"]
    print('equation:', eq)

    if filtered_df is None:
        filtered_df = copy.deepcopy(df)
        print('Length of dataframe:', len(filtered_df))
    else:
        filtered_df = filtered_df
        print('Length of dataframe:', len(filtered_df))

    # Split the equation into individual components
    components = eq.split()
    print('equation:', components)

    # Extract the new column name
    new_column_name = components[0][1:-1]
    print('new column:', new_column_name)

    # Initialize the result column
    result_column = None
    intermediate_result = None

    for i in range(2, len(components), 2):
        operator = components[i - 1]
        operand = components[i]

        if operand.startswith("'") and operand.endswith("'"):
            # Operand is a column name
            column_name = operand[1:-1]
            operand_column = filtered_df[column_name]
        else:
            operand_column = float(operand)
        
        if operator in ['=', '+', '-', '*', '/']:
            if operator == '=':
                intermediate_result = copy.copy(operand_column)
            elif operator == '+':
                intermediate_result += operand_column
                print('operand_column:', operand_column)
            elif operator == '-':
                intermediate_result -= operand_column
                print('operand_column:', operand_column)
            elif operator == '*':
                intermediate_result *= operand_column
                print('operand_column:', operand_column)
            elif operator == '/':
                intermediate_result /= operand_column
                print('operand_column:', operand_column)
            # Append the new column to the DataFrame
            result_column = intermediate_result
            filtered_df[new_column_name] = result_column
        else:
            if operator == '<':
                filtered_df = filtered_df[filtered_df[column_name] < operand_column]
            elif operator == '>':
                filtered_df = filtered_df[filtered_df[column_name] > operand_column]
            elif operator == '==':
                filtered_df = filtered_df[filtered_df[column_name] == operand_column]
    response = {
        "columns": [
            {
                "Header": column,
                "accessor": column
            }
            for column in filtered_df.columns
        ],
        "data": format_dataframe(filtered_df.head(100))
    }

    return jsonify(response)

def format_dataframe(df):
    formatted_data = []
    for _, row in df.iterrows():
        formatted_row = {}
        for column, value in row.items():
            if isinstance(value, pd.Timestamp):
                formatted_row[column] = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, bool):
                formatted_row[column] = str(value)
            else:
                formatted_row[column] = value
        formatted_data.append(formatted_row)
    return formatted_data

@app.route("/chartData", methods=["POST"])
def createTable():
    
    formattedData = format_frame()
    return jsonify(formattedData)

def format_frame():
    global groupedData, pie_groupedData, state_1
    data = request.json
    xAxisParam_1 = data['xAxisParam']
    yAxisParams_1 = data['yAxisParams']
    type_1 = data['type']
    grouped_data = None

    headers = []

    if len(xAxisParam_1) == 0:  
        headers.append('count')
        for yAxisParm in yAxisParams_1:
            if yAxisParm:
                headers.append(yAxisParm)
        
    else:
        headers.append(xAxisParam_1)
        for yAxisParm in yAxisParams_1:
            if yAxisParm:
                headers.append(yAxisParm)
        grouped_data = pd.DataFrame.from_dict(groupedData, orient='index').reset_index()
        grouped_data.columns = headers
    print('headers:', headers)

    frame_rep_1 = dict()
    frame_rep_1["columns"] = [{
        "Header": column,
        "accessor": column
    } for column in headers]
    frame_rep_1["data"] = []

    if len(xAxisParam) == 0:
        if pie_groupedData is not None:
            state_1["frame"] = pie_groupedData.head(200)
            print('state_1:', state_1["frame"])
            for _, row in state_1["frame"].iterrows():
                formatted_row = {}
                for column, value in row.items():               
                    formatted_row[column] = value
                frame_rep_1["data"].append(formatted_row)
            
    else:
        if grouped_data is not None:
            state_1["frame"] = grouped_data.head(200)
            print('state_1:', state_1["frame"])
            for _, row in state_1["frame"].iterrows():
                formatted_row = {}
                for column, value in row.items():               
                    formatted_row[column] = value
                frame_rep_1["data"].append(formatted_row)
    
    print('frame_rep_1:', frame_rep_1)

    return frame_rep_1

@app.route("/summarystat", methods=["POST"])
def statcalc():
    global df
    data = request.json
    print('Received Data:', data)
    stat = data['value']
    series = data['SeriesOption']

    if stat == 'count':
        result = df.shape[0]
    elif stat == 'median':
        result = df[series].median().round(2)
    elif stat == 'average':
        result = df[series].mean().round(2)
    elif stat == 'sum':
        result = df[series].sum()
    else:
        result = df[series].nunique()
    
    response = jsonify({f"{series} ({stat})": result})
    print('response:', response)
    return response
    
if __name__ == "__main__":
    app.run(port=5000)