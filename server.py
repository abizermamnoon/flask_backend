from flask import Flask, request, send_file, make_response, Response, jsonify
from flask_cors import CORS
from io import StringIO
import json
import os
import datetime
import csv
import time
import numpy as np
import copy
import pandas as pd
from itertools import chain

app = Flask(__name__)
CORS(app)

uploadedFileName = None
uploadedFileType = None
df = None
df_0 = None
df_1 = None
df_2 = None
df_3 = None
df_4 = None
df_5 = None
df_6 = None 
df_7 = None
nondf_list = []
df_list1 = []
cols = []
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
box_groupedData = None
groupedData = {}
grouped = None
sumstat = {}
heat_groupedData = None
defdf_list = []
dtypes_dict = {}
uploaded_files_info = []

@app.route("/upload", methods=["POST"])
def upload():
    start_time = time.time()

    global df, df_0, df_1, df_2, df_3, df_4, cols, df_5, df_6, df_7, nondf_list, defdf_list, uploaded_files_info

    nondf_list = [df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7]
    
    files_dict = request.files.to_dict(flat=False)
    files_list = list(files_dict.keys())
    # print('list of files:', files_list)

    for index, item in enumerate(files_list):
        nondf_list[index] = pd.read_csv(request.files[item])
        
    defdf_list = [item for item in nondf_list if item is not None]
    # print('defdf_list:', len(defdf_list))

    for key, file_list in files_dict.items():
        if file_list:  # Check if the file list is not empty
            uploaded_file_name = file_list[0].filename
            uploaded_file_type = file_list[0].content_type

            file_name_without_extension = os.path.splitext(uploaded_file_name)[0]

            file_info = {
                "file": file_name_without_extension,
                "path": f"/{uploaded_file_name}",
                "type": uploaded_file_type,
            }

            uploaded_files_info.append(file_info)
            
    if len(defdf_list) == 1:
        df = defdf_list[0]
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    return {"files": uploaded_files_info}

@app.route('/findtable', methods=["POST"])
def findtab():
    data = request.json
    print('Received Data:', data)
    global defdf_list
    print('length of df list:', len(defdf_list))
    selectedtab = data['selectedFiles']
    print('selectedtab:', selectedtab)
    formattedData = format_Frame(defdf_list[selectedtab])
    return jsonify(formattedData)

def format_Frame(df):
    state = {}
    state = {
        "frame": None
    }
    headers = df.columns

    frame_rep = dict()
    frame_rep["columns"] = [{
        "Header": column,
        "accessor": column
    } for column in headers]
    frame_rep["data"] = []

    state["frame"] = df.head(10)
    # print('frame_rep:', frame_rep)
    frame_rep["data"] = []
    for _, row in state["frame"].iterrows():
        formatted_row = {}
        for column, value in row.items():
            if isinstance(value, pd.Timestamp):
                formatted_row[column] = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, bool):
                formatted_row[column] = str(value)
            elif pd.isna(value):  # Check for NaN values
                formatted_row[column] = "N/A"
            else:
                formatted_row[column] = value
        frame_rep["data"].append(formatted_row)
    
    # print('frame_rep:', frame_rep)
    return frame_rep


@app.route("/jointable", methods=["POST"])
def join():
    data = request.json
    print('Received Data:', data)
    join = data['joinType']
    primary_key = data['primaryKey']
    
    def perform_join(how):
        global df, defdf_list, df, nondf_list, uploaded_files_info
        colnam = [] 
        df = None
        for index, item in enumerate(defdf_list):
            file_name_without_extension = uploaded_files_info[index]["file"]
            if defdf_list[index] is not None:  # Skip if DataFrame is None
                temp_df = defdf_list[index].copy() 
                for col in temp_df.columns:
                    if df is not None:
                        if col != primary_key:
                            col_with_prefix = f"{file_name_without_extension}_{col}"
                            colnam.append(col_with_prefix)
                    else:
                        col_with_prefix = f"{file_name_without_extension}_{col}"
                        colnam.append(col_with_prefix)
                if df is None:
                    df = temp_df
                else:

                    df = pd.merge(df, temp_df, on=primary_key, how=how)
        print('colnam:', colnam)
        print('number of columns:', len(colnam))
        df.columns = colnam
    
        print('columns in df before dropping duplicates:', df.columns)
        print('number of columns in dataframe:', len(df.columns))

        # Drop duplicate columns
        # df = df.loc[:, ~df.columns.duplicated()]
        # print('columns in df after dropping duplicates:', df.columns)

        for index, item in enumerate(nondf_list):
            if item is None:
                nondf_list[index] = df
                break
        defdf_list = [item for item in nondf_list if item is not None]
        print('Number of dataframes:', len(defdf_list))

        return df.columns.tolist()

    if join == 'inner':
        return perform_join('inner')
    if join == 'left':
        return perform_join('left')
    if join == 'right':
        return perform_join('right')
    
@app.route("/dropcol", methods=["POST"])
def dropcol():
    global df, defdf_list
    data = request.json
    print('Received Data:', data)
    selectedTab = data['selectedFiles']
    yAxisParams = []
    for yAxisParm in data['yAxisParams']:
        if yAxisParm:
            yAxisParams.append(yAxisParm)
    print('yAxisParams:', yAxisParams)
    defdf_list[selectedTab] = defdf_list[selectedTab].drop(columns=yAxisParams, axis=1 )
    print('df columns:', defdf_list[selectedTab].columns)
    return 'working'

@app.route("/nullval", methods=["POST"])
def countnul():
    global df, defdf_list
    print('length of df list:', len(defdf_list))

    data = request.json
    print('Received Data:', data)
    selectedtab = None
    if data['selectedFiles'] is not None:
        selectedtab = data['selectedFiles']
    
    if selectedtab is not None:
        print('selectedtab:', selectedtab)
        df = defdf_list[selectedtab]

    if df is not None:
        df_nul = df.isnull().sum()
        df_nul = df_nul.to_frame(name='count_nuls').reset_index()
        df_nul.columns = ['columns', 'count_nuls']
        df_nul = df_nul[df_nul['count_nuls'] > 0]
        # print('df_nul:', df_nul)
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
    # print('state:', state["frame"])
    for _, row in state["frame"].iterrows():
        formatted_row = {}
        for column, value in row.items():               
            formatted_row[column] = value
        frame_rep["data"].append(formatted_row)
    
    # print('frame_rep:', frame_rep)

    return frame_rep

@app.route("/dropna", methods=["POST"])
def findna():
    data = request.json
    action = data['action']

    print('Requested Data:', data)
    selectedTab = data['selectedFiles']

    if action == 'drop':
        dropna(selectedTab)
        return "Rows with NA values have been dropped"
    if action == 'next':
        nextna(selectedTab)
        return "cells with NA values have been replaced by rows below it"
    if action == 'prev':
        prevna(selectedTab)
        return "cells with NA values have been replaced by rows above it"
    if action == 'interp':
        interpna(selectedTab)
        return "cells with NA values have been replaced by linear interpolation"
    
def dropna(selectedTab):
    global df, defdf_list

    defdf_list[selectedTab] = defdf_list[selectedTab].dropna()

def nextna(selectedTab):
    global df, defdf_list
    defdf_list[selectedTab] = defdf_list[selectedTab].fillna(method ='bfill')

def prevna(selectedTab):
    global df, defdf_list
    defdf_list[selectedTab] = defdf_list[selectedTab].fillna(method ='pad')

def interpna(selectedTab):
    global df, defdf_list
    defdf_list[selectedTab] = defdf_list[selectedTab].interpolate(method ='linear', limit_direction ='forward')


@app.route("/loadTable", methods=["POST"])
def load():

    global df, grouped_monthly, grouped_yearly, grouped_daily, state, frame_rep, response, grouped_data, data_types, sumstat, defdf_list

    data = request.json
    print('Received data:', data)
    selectedTab = data['selectedFiles']

    df = defdf_list[selectedTab]

    start_time = time.time()

    state["frame"] = df.head(100)
    frame_rep = formatFrame()
    response = jsonify(frame_rep)

    # Perform groupby operations immediately after uploading the file for all columns
    
    for column in df.columns:
        grouped_data[column] = df.groupby(column)
        grouped_data[column + "_group_length"] = len(grouped_data[column])
    
    for column in df.columns:
        print('column type:', df[column].dtype)
        if df[column].dtype in ['int64', 'float64']:
            statistics = df[column].describe()
            sumstat[column] = {
            'min': statistics['min'],
            '25%': statistics['25%'],
            '50%': statistics['50%'],
            '75%': statistics['75%'],
            'max': statistics['max']
            }
    print('sumstat:', sumstat)
        
    # Slice the first 10 rows and save it as '10_rows.json'
    slicedData = df.head(5)
    slicedDataFilePath = "./public/10_rows.json"
    slicedData.to_json(slicedDataFilePath, orient="records")

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    return "Data has been loaded"

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
            if isinstance(value, bool):
                formatted_row[column] = str(value)
            else:
                formatted_row[column] = value
        frame_rep["data"].append(formatted_row)
    print("Frame formatted")
    return frame_rep

@app.route("/transformtable", methods=["POST"])
def transform():
    global df, filtered_df, grouped_monthly, grouped_yearly, grouped_daily, grouped_data, response
    df = None
    df = filtered_df
    state = {}
    state = {
        "frame": None
    }
    frame_rep = {} 
    response = {}
    grouped_data = {}

    start_time = time.time()

    for column in df.columns:
        grouped_data[column] = df.groupby(column)
        grouped_data[column + "_group_length"] = len(grouped_data[column])

    state["frame"] = df.head(100)
    frame_rep = formatFrame()
    response = jsonify(frame_rep)

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
            grouped_monthly_data[column + "_group_length"] = len(grouped_monthly_data[column])
        return grouped_monthly_data
    else:
        return None

def group_by_yearly():
    global df, converted_columns
    if len(converted_columns) > 0:
        grouped_yearly_data = {}
        for column in converted_columns:
            grouped_yearly_data[column] = df.groupby(df[column].dt.strftime('%Y'))
            grouped_yearly_data[column + "_group_length"] = len(grouped_yearly_data[column])
        return grouped_yearly_data
    else:
        return None

def group_by_daily():
    global df, converted_columns
    if len(converted_columns) > 0:
        grouped_daily_data = {}
        for column in converted_columns:
            grouped_daily_data[column] = df.groupby(df[column].dt.strftime('%Y-%m-%d'))
            grouped_daily_data[column + "_group_length"] = len(grouped_daily_data[column])
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
            if isinstance(value, bool):
                formatted_row[column] = str(value)
            else:
                formatted_row[column] = value
        frame_rep["data"].append(formatted_row)
    print("Frame formatted")
    return frame_rep

@app.route("/")
def index():

    filePath = "./public/10_rows.json"  # Change this to dynamic file path if needed
    headers = {}

    # Set the appropriate response headers for the compressed content
    headers["Content-Type"] = "application/json"
    
    response = make_response(send_file(filePath))
    response.headers = headers

    return response

@app.route("/coltype", methods=["POST"])
def findColType():
    global df, dtypes_dict
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
    global df, grouped_monthly, grouped_daily, grouped_yearly, grouped_data, converted_columns, xAxisParam, yAxisParams, xAxisParam_1, yAxisParams_1, type_1, type, box_groupedData, sumstat, heat_groupedData, dtypes_dict, grouped
    min_yAxisData = 0
    max_yAxisData = 0
    print('dtypes_dict:', dtypes_dict)

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

    if len(xAxisParam) == 0 and type != 'boxplot':
        pie_groupedData = df.groupby(yAxisParams[0]).size().reset_index(name='count')
        pie_groupedData = pie_groupedData.sort_values('count', ascending=False)
        sortedData = {
            "xAxisData": [''.join(map(str, group)) for group in pie_groupedData[yAxisParams].values],
            "yAxisData": pie_groupedData['count'].values.tolist()
        }
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        return sortedData, 200
    elif type == 'boxplot':
        groupedData = {}
        result = []
        for param in yAxisParams:
            grouped = sumstat[param]
            valuesArray = list(grouped.values())
            result.append(valuesArray)
        return result 
    elif type != 'heatmap' and len(xAxisParam) > 0:
        
        if xAxisParam not in converted_columns:
           groupedData = {}
           counter = 0
           grouped = grouped_data[xAxisParam]
           num_groups = grouped_data[xAxisParam + "_group_length"]
           print('length of grouped:', num_groups)

           for groupKey, group in grouped:
                if num_groups > 100000:
                    if counter % 1000 == 0:  # Check if the counter is a multiple of 100
                        first_values = group.head(1)
                        for _, entry in first_values.iterrows():
                            groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
                    counter += 1
                elif num_groups > 10000:
                    if counter % 100 == 0:  # Check if the counter is a multiple of 100
                        first_values = group.head(1)
                        for _, entry in first_values.iterrows():
                            groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
                    counter += 1
                elif num_groups > 1000:
                    if counter % 10 == 0:  # Check if the counter is a multiple of 100
                        first_values = group.head(1)
                        for _, entry in first_values.iterrows():
                            groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
                    counter += 1
                else:
                    first_values = group.head(1)
                    for _, entry in first_values.iterrows():
                        groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
        
        else:
            groupedData = {}
            counter = 0
            if interval == "daily":
                grouped = grouped_daily[xAxisParam]
                num_groups = grouped_daily[xAxisParam + "_group_length"]
                print('length of grouped:', num_groups)              
            elif interval == "monthly":
                grouped = grouped_monthly[xAxisParam]  
                num_groups = grouped_monthly[xAxisParam + "_group_length"]
                print('length of grouped:', num_groups)            
            elif interval == "yearly":
                grouped = grouped_yearly[xAxisParam]  
                num_groups = grouped_yearly[xAxisParam + "_group_length"]
                print('length of grouped:', num_groups)           
            else:
                return {"msg": "Invalid interval"}, 400
            
            for groupKey, group in grouped:
                if num_groups > 100000:
                    if counter % 1000 == 0:  # Check if the counter is a multiple of 100
                        first_values = group.head(1)
                        for _, entry in first_values.iterrows():
                            groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
                    counter += 1
                elif num_groups > 10000:
                    if counter % 100 == 0:  # Check if the counter is a multiple of 100
                        first_values = group.head(1)
                        for _, entry in first_values.iterrows():
                            groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
                    counter += 1
                elif num_groups > 1000:
                    if counter % 10 == 0:  # Check if the counter is a multiple of 100
                        first_values = group.head(1)
                        for _, entry in first_values.iterrows():
                            groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
                    counter += 1
                else:
                    first_values = group.head(1)
                    for _, entry in first_values.iterrows():
                        groupedData[groupKey] = {yAxisParam: entry[yAxisParam] if yAxisParam in entry else None for yAxisParam in yAxisParams}
            
        sortedData = {
            "xAxisData": sorted(groupedData.keys()),  # Sort the keys
            "yAxisData": [[groupedData[key][yAxisParam] for yAxisParam in yAxisParams] for key in sorted(groupedData.keys())]
        }
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        return sortedData, 200
    
    else:
        heat_groupedData = pd.DataFrame(columns=[xAxisParam] + yAxisParams)
        if interval == "daily":
            grouped = grouped_daily[xAxisParam]
        elif interval == "monthly":
            grouped = grouped_monthly[xAxisParam]
        elif interval == "yearly":
            grouped = grouped_yearly[xAxisParam]           
        else:
            return {"msg": "Invalid interval"}, 400
        
        if dtypes_dict[yAxisParams[0]] == 'int':
            for groupKey, group in grouped:
                first_values = group.head(1)
                if not first_values.empty:
                    values_dict = {yAxisParam: first_values.iloc[0][yAxisParam] if yAxisParam in first_values.columns else None for yAxisParam in yAxisParams}
                    heat_groupedData = heat_groupedData.append({xAxisParam: groupKey, **values_dict}, ignore_index=True)
                # print('heat_groupedData:', heat_groupedData)
        else:
            print('error')
            return

        heat_groupedData[xAxisParam] = pd.to_datetime(heat_groupedData[xAxisParam])

        if interval == 'monthly':
            heat_groupedData['month'] = heat_groupedData[xAxisParam].dt.month
            heat_groupedData[xAxisParam] = heat_groupedData[xAxisParam].dt.year
            heat_groupedData['datetime_uniqueness'] = heat_groupedData[xAxisParam].factorize()[0]
            heat_groupedData['month_uniqueness'] = heat_groupedData['month'].factorize()[0]
            heat_groupedData['index_array'] = heat_groupedData.apply(lambda row: [row['datetime_uniqueness'], row['month_uniqueness'], row[yAxisParams[0]]], axis=1)
            min_yAxisData = heat_groupedData[yAxisParams[0]].min()
            max_yAxisData = heat_groupedData[yAxisParams[0]].max()
            heat_groupedData = heat_groupedData.drop(['datetime_uniqueness', 'month_uniqueness'], axis=1)
        elif interval == 'daily':  
            heat_groupedData['day'] = heat_groupedData[xAxisParam].dt.strftime('%d')
            heat_groupedData[xAxisParam] = heat_groupedData[xAxisParam].dt.strftime('%Y-%m')
            heat_groupedData['datetime_uniqueness'] = heat_groupedData[xAxisParam].factorize()[0]
            heat_groupedData['day_uniqueness'] = heat_groupedData['day'].factorize()[0]
            heat_groupedData['index_array'] = heat_groupedData.apply(lambda row: [row['datetime_uniqueness'], row['day_uniqueness'], row[yAxisParams[0]]], axis=1)
            min_yAxisData = heat_groupedData[yAxisParams[0]].min()
            max_yAxisData = heat_groupedData[yAxisParams[0]].max()
            heat_groupedData = heat_groupedData.drop(['datetime_uniqueness', 'day_uniqueness'], axis=1)

        # Calculate min and max values of the yAxisData
        # print('heat_groupedData:', heat_groupedData)
        # print('min:', min_yAxisData)
        # print('max:', max_yAxisData)

        second_column = heat_groupedData.columns[2]

         # Create the sortedData dictionary as per the desired format
        sortedData = {
            "xAxisData": heat_groupedData[xAxisParam].unique().tolist(),
            "yAxisData": heat_groupedData[second_column].unique().tolist(),
            "data": heat_groupedData['index_array'].tolist(),
            "min": min_yAxisData,
            "max": max_yAxisData,
            "yParam": heat_groupedData[yAxisParams[0]].tolist(),
        }

        # Convert the sortedData dictionary to JSON format
        sortedData_json = json.dumps(sortedData)

        # print('sorted data:', sortedData_json)

        return sortedData_json, 200
        
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
        # print('Length of dataframe:', len(filtered_df))
    else:
        filtered_df = filtered_df
        # print('Length of dataframe:', len(filtered_df))

    # Split the equation into individual components
    components = eq.split()
    print('equation:', components)

    if len(components) >= 3 and components[:2] == ['fmt', '=']:
        fmt = ' '.join(components[2:])
        print('fmt:', fmt)
        filtered_df = convert_datetime_columns(filtered_df, fmt)
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

def convert_datetime_columns(df, fmt):
    datetime_columns = df.select_dtypes(include=[object]).columns
    global converted_columns
    print('datetime_columns:', datetime_columns)
    # print('format:', fmt)
    for column in datetime_columns:
        try:
            df[column] = pd.to_datetime(df[column], format=fmt)
            print('format:', fmt)
            converted_columns.append(column)
            print('converted_columns:', converted_columns)
        except ValueError:
            pass
    print('converted_columns:', converted_columns)
    return df

@app.route("/chartData", methods=["POST"])
def createTable():
    
    formattedData = format_frame()
    return jsonify(formattedData)

def format_frame():
    global groupedData, pie_groupedData, state_1, heat_groupedData, grouped
    data = request.json
    xAxisParam_1 = data['xAxisParam']
    yAxisParams_1 = data['yAxisParams']
    type_1 = data['type']
    interval_1 = data['interval']
    grouped_data = pd.DataFrame()
    print('Chart Type1:', type_1)
    print('Data Received:', data)
    dtype_list = [] 

    for yAxisParm in yAxisParams_1:
        if yAxisParm:
            dtype_list.append(df[yAxisParm].dtype)
    print('dtypes list:', dtype_list)

    headers = []
    if len(xAxisParam_1) == 0:
        if np.object_ in dtype_list:
            headers.append('count')
            for yAxisParm in yAxisParams_1:
                if yAxisParm:
                    headers.append(yAxisParm)
        else:
            print('in boxplot')
            headers.append('stats')
            for yAxisParm in yAxisParams_1:
                if yAxisParm:
                    headers.append(yAxisParm)
                    column_data = df[yAxisParm].describe()[['min', '25%', '50%', '75%', 'max']]
                    grouped_data = grouped_data.append(column_data)
            grouped_data = grouped_data.transpose().reset_index()
            grouped_data.columns = headers   
            # print('grouped_data:', grouped_data)
        
    elif len(xAxisParam_1) > 0:
        if heat_groupedData is None:
            headers.append(xAxisParam_1)
            for yAxisParm in yAxisParams_1:
                if yAxisParm:
                    headers.append(yAxisParm)
            grouped_data = pd.DataFrame.from_dict(groupedData, orient='index').reset_index()
            grouped_data.columns = headers
        else:
            if 'index_array' in heat_groupedData.columns:
                heat_groupedData = heat_groupedData.drop(['index_array'], axis=1)
            headers.append(xAxisParam_1)
            for yAxisParm in yAxisParams_1:
                if yAxisParm:
                    headers.append(yAxisParm)
            if interval_1 == 'monthly':
                headers.append('month')
            if interval_1 == 'daily':
                headers.append('day')
            

    frame_rep_1 = dict()
    frame_rep_1["columns"] = [{
        "Header": column,
        "accessor": column
    } for column in headers]
    frame_rep_1["data"] = []

    if len(xAxisParam) == 0: 
        if np.object_ in dtype_list:
            if pie_groupedData is not None:
                state_1["frame"] = pie_groupedData.head(200)
                # print('state_1:', state_1["frame"])
                for _, row in state_1["frame"].iterrows():
                    formatted_row = {}
                    for column, value in row.items():               
                        formatted_row[column] = value
                    frame_rep_1["data"].append(formatted_row)
        else:
            if grouped_data is not None:
                print('type of frame')
                state_1["frame"] = grouped_data
                # print('state_1:', state_1["frame"])
                for _, row in state_1["frame"].iterrows():
                    formatted_row = {}
                    for column, value in row.items():               
                        formatted_row[column] = value
                    frame_rep_1["data"].append(formatted_row)
            
    elif len(xAxisParam) > 0 and type_1 != 'boxplot':
        if heat_groupedData is None:
            state_1["frame"] = grouped_data.head(200)
            # print('state_1:', state_1["frame"])
            for _, row in state_1["frame"].iterrows():
                formatted_row = {}
                for column, value in row.items():               
                    formatted_row[column] = value
                frame_rep_1["data"].append(formatted_row)
        if heat_groupedData is not None:
            state_1["frame"] = heat_groupedData.head(200)
            heat_groupedData = None
            # print('state_1:', state_1["frame"])
            for _, row in state_1["frame"].iterrows():
                formatted_row = {}
                for column, value in row.items():               
                    formatted_row[column] = value
                frame_rep_1["data"].append(formatted_row)
    # print('frame_rep_1:', frame_rep_1)

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