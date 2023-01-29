from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the API"

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    result = process_data(data)
    return result

def process_data(data):
    # Do some processing on the data
    result = data + " processed"
    return result

if __name__ == '__main__':
    app.run(debug=True)
