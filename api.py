from flask import Flask, request, jsonify
from flask_restful import Api
from datetime import datetime
import api_college_pred, api_major_pred
app = Flask(__name__)
api = Api(app)

@app.route('/')
def hello():
    if 'name' in request.args:
        name = request.args['name']
    else:
        name = 'New friend'

    return """
         <html><body>
             <h1>Hello, {0}!</h1>
             The time is {1}.
         </body></html>
         """.format(name, str(datetime.now()))


@app.route('/school_predict/<int:s1>/<int:s2>/<int:s3>/<int:s4>')
def school_predict(s1, s2, s3, s4):
    if (s1 > 600 or s1 < 0) or (s2 > 600 or s2 < 0) or (s3 > 600 or s3 < 0) or (s4 > 30  or s4 < 0):
        return 'Error: Invalid input. Score should be in the correct range.'
    score_list = [s1, s2, s3, s4]
    dis = api_college_pred.prediction_school(score_list)[0]
    return jsonify({'school_id': dis})

@app.route('/major_predict/<int:s1>/<int:s2>/<int:s3>')
def major_predict(s1, s2, s3):
    if (s1 > 600 or s1 < 0) or (s2 > 600 or s2 < 0) or (s3 > 600 or s3 < 0):
        return 'Error: Invalid input. Score should be in the range of [0,600].'
    score_list1 = [s1, s2, s3]
    dis = api_major_pred.major_recommendation(score_list1)
    return jsonify({'major': dis})

if __name__ == '__main__':
    app.run(debug = True)