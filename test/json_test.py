from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/test', methods=['POST'])
def test():
    try:
        input_data = request.get_json()
        print(input_data)
        result_data = {"test": 'test info from python'}
        return jsonify(result_data)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=11200)
