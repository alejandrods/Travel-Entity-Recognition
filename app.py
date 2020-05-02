#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import flask
import logging
from json import loads
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv

from pred import prediction


load_dotenv()

app = flask.Flask(__name__)
CORS(app)
app.config['PROPAGATE_EXCEPTIONS'] = True

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s :: %(levelname)s :: %(message)s')

PORT = os.environ.get('PORT', 7000)


@app.route('/')
@cross_origin()
def index():
    return flask.render_template('main.html')


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    income_query = loads(flask.request.data)['input']

    print("B:", income_query)

    logging.info('Query received: {}'.format(income_query))

    res = prediction(income_query)

    logging.info('Result: {}'.format(res))

    return flask.jsonify({
        "version": "v0",
        "pred": res
    })


@app.route('/health_liveness')
@cross_origin()
def healthcheck():
    """
    Check status code
    :return:
    """
    return json.dumps({'status': 'success'}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0',
            debug=True,
            port=PORT)
