import os.path
from datetime import datetime
from flask import Flask, request, jsonify
import base64
from src.javaweb.utils.data_preprocess import load_data
from src.javaweb.utils.classifier import classifier_model

app = Flask(__name__)

file_path = 'data/java/'


def classify(file_name=str):
    signal = load_data(file_name)
    return classifier_model(signal, 'models/202311301551_96.pth')
    pass


def save_data(file_name, file_data):
    print('Save data...')
    decoder_data = base64.b64decode(file_data)
    with open(file_path + file_name, 'wb') as file:
        file.write(decoder_data)
    print('Data is saved in ' + file_path + file_name)
    pass


def parse_filename(file_name):
    file_name = os.path.splitext(file_name)[0]
    parts = file_name.split('_')
    date_str, time_str, id_str = parts[:3]
    date_format = "%Y%m%d"
    time_format = "%H%M%S"
    date = datetime.strptime(date_str, date_format)
    time = datetime.strptime(time_str, time_format).time()

    return date.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S'), id_str
    pass


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
