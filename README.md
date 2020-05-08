# Travel Entity Recognition Travel-Related Sentences

Project to create a named entity recognition (NER) for travel-related sentences. 
I developed this project inspired by how Gmail extracts information from the 
travel-related emails and automatically inserts the event in your calendar. 
It developed it using Keras and the model is deployed as REST API.

### Table of Contents  

[Installation](#Installation) 

[Train](#Train)  

[Run App](#Deploy)  

[Examples](#Examples)  

<a name="Installation"></a>
#### Installation

1- Clone the repository in your local machine:
```
git clone git@github.com:alejandrods/Travel-Entity-Recognition.git
```

2- Install requirements
```
pip install -r requirements.txt
```

> Change `tensorflow-gpu` to `tensorflow` in `requirements.txt` if you are not available to use GPU.

3- Environment variables required

```
DATA_PATH (Path to data - i.e: ./data)
CONVERT_PATH (Path to converted files - i.e: ./converted)
DATASET_FILE (Dataset file - i.e: travel_set.csv)
QUERY_FILE (Name for query converted file  - i.e: query_set.txt)
LABEL_FILE (Name for label converted file - i.e: labels_set.txt)
GLOVE_DIR= (Path to Glove embeddings - i.e: ./embedding/glove.6B.100d.txt)
EMBEDDING_DIM (Embedding Dimension - i.e: 100)
MAX_SEQ_LEN (Max length sequences - i.e: 60)
MODEL_DIR (Path to model dir - i.e: ./model)
```

4- Download pre-trained words vector in `.txt` format from [this site - Glove](https://nlp.stanford.edu/projects/glove/)

<a name="Train"></a>
### Train

1- Set environment variables in `.env`

2- To train a new model, run the next command:

`python train.py`

<a name="Deploy"></a>
### Run App

1- Set environment variables in `.env`

2- To deploy the front-end using flask-app, run the next command:
`python app.py`

<a name="Examples"></a>
### Examples
I need a flight from New York to Barcelona on may 4th morning
![](./images/example1.PNG)

What is the status of the flight UX 1092
![](./images/example2.PNG)