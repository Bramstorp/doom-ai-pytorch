from flask import Flask, render_template, url_for, request
import sys
sys.path.append('../')

from doom_ai import run_ai


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/testing',methods=['GET'])
def testing_get():
    return render_template("testing.html")

@app.route('/testing',methods=['POST'])
def testing_post():
    return run_ai(True)

@app.route('/training',methods=['GET'])
def training_get():
    return render_template("training.html")

@app.route('/training',methods=['POST'])
def training_post():
    return run_ai()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)