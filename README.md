<h1><img height="60" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/UnIT_logo.png">&nbsp;<img height="60" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/logo.png">&nbsp;UnIT extended 2019  &nbsp ItAintMuchButItsHonestWork</h1>

You can acces UnIT extended 2019 hackaton assignment [here](https://github.com/LachubCz/ItAintMuchButItsHonestWork/blob/master/assignment.pdf). This repository contains source code created by group ItAintMuchButItsHonestWork. Based on tournament metrics we've reached succes rate of 76.89 % in recognizing ellipses and won first place.

### Pipeline

<img height="579" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/pipeline.PNG">

Usage
-----
**Application**
##### python3 src/main.py [--mode mode] [--csv-input csv-input] [--csv-output csv-output] [--images-path images-path] [--ground-truths-path ground-truths-path]
###### Parameters:

    --mode mode                              |  application mode (eval, entry)
    --csv-input csv-input                    |  filename of input csv file containing training data
    --csv-output csv-output                  |  filename of output csv file containing competition data
    --images-path images-path                |  image folder path
    --ground-truths-path ground-truths-path  |  ground truths folder path

**Data generator**
##### python3 src/batch.py [--batches batches]
###### Parameters:

    --batches batches  |  number of generated batches of size 32

**Classifier training**
##### python3 src/classification.py [--trn-data-size trn-data-size] [--model-name model-name]
###### Parameters:

    --trn-data-size trn-data-size  |  number of images for training
    --model-name model-name        |  filename of SVM model

****
###### Created by: Petr Buchal, Martin Ivančo, Vladimír Jeřábek
