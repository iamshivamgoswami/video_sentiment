from flask import Flask, request, jsonify
from sentiment_analysis import sentiment_
app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text')
    
    # Perform some processing on the text (for now, just return the input text)
    
    
    return jsonify({'processed_text': sentiment_(text)})

if __name__ == '__main__':
    app.run(debug=True)
