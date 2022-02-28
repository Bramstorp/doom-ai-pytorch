from flask import Flask, render_template, url_for, request
import test 

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/testing',methods=['POST', 'GET'])
def testing():
    return render_template("testing.html")

@app.route('/training',methods=['POST', 'GET'])
def training():
    return render_template("training.html")

@app.route('/result',methods=['POST', 'GET'])
def result():
    return test.test()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)