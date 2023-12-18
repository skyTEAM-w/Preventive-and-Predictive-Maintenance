import base64
import os.path
import pymysql
from datetime import datetime

from flask import Flask, request, jsonify

from src.javaweb.utils.classifier import classifier_model
from src.javaweb.utils.data_preprocess import load_data
from src.javaweb.utils.predict import *

app = Flask(__name__)

file_path = 'data/java/'
models_dir = 'models/'


def classify(file_name=str):
    """
    对文件进行分类。

    Args:
        file_name (str): 文件名。

    Returns:
        object: 分类结果。
    """
    signal = load_data(file_name)
    return classifier_model(signal, 'models/202311301551_96.pth')


def predict(file_name=str):
    sensors_data = load_data_rul(file_path + file_name,
                                 models_dir + 'kmeans_model.joblib',
                                 models_dir + 'scaler_params.pkl')
    return predict_rul(sensors_data, models_dir + '202312181709_001.pth')


def save_data(file_name, file_data):
    """
    保存文件数据。

    Args:
        file_name (str): 文件名。
        file_data (str): 文件数据。
    """
    print('保存数据...')
    decoder_data = base64.b64decode(file_data)
    with open(file_path + file_name, 'wb') as file:
        file.write(decoder_data)
    print('数据已保存在 ' + file_path + file_name)


def parse_filename(file_name):
    """
    解析文件名，提取日期、时间和 ID。

    Args:
        file_name (str): 文件名。

    Returns:
        tuple: 包含日期、时间和 ID 的元组。
    """
    file_name = os.path.splitext(file_name)[0]
    parts = file_name.split('_')
    date_str, time_str, id_str = parts[:3]
    date_format = "%Y%m%d"
    time_format = "%H%M%S"
    date = datetime.strptime(date_str, date_format)
    time = datetime.strptime(time_str, time_format).time()

    return date.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S'), id_str


@app.route('/app', methods=['POST'])
def application_serve():
    try:
        input_data = request.get_json()
        file_data = input_data['file_data']
        file_name = input_data['file_name']
        save_data(file_name, file_data)
        classify_result = classify(file_name)
        print(type(classify_result))
        date, time, id = parse_filename(file_name)
        result_data = {
            'id': id,
            'date': date,
            'time': time,
            'result-id': str(classify_result)
        }
        print(result_data)
        return jsonify(result_data)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # classify('received_file.txt')
    app.run(host='127.0.0.1', port=11200)
    # print(predict('rul_test.csv'))