#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
from dotenv import load_dotenv
import flask

from pred import prediction
load_dotenv()

app = flask.Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s :: %(levelname)s :: %(message)s')

PORT = os.environ.get('PORT', 7000)


@app.route('/healthcheck')
def healthcheck():
    """
    Check status code
    :return:
    """
    return json.dumps({'status': 'success'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    body = json.loads(flask.request.data)
    query = body['query']

    logging.info('Query received: {}'.format(query))

    res = prediction(query)

    logging.info('Result: {}'.format(res))

    response = app.response_class(
        response=json.dumps(res),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0',
            debug=True,
            port=PORT)
