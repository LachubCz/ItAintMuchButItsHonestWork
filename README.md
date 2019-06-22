<h1><img height="60" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/UnIT_logo.png">&nbsp;<img height="60" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/logo.png">&nbsp;UnIT extended 2019 ItAintMuchButItsHonestWork</h1>

You can acces UnIT extended 2019 hackaton assignment [here](https://github.com/LachubCz/ItAintMuchButItsHonestWork/blob/master/assignment.pdf). This repository contains source code created by group ItAintMuchButItsHonestWork. Based on tournament metrics we've reached succes rate of 76.89 % in recognizing ellipses and won first place.

### Pipeline

<img height="579" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/pipeline.PNG">

Usage
-----
**Application**
##### python3 src/main.py [--mode mode] [--csv-input csv-input] [--csv-output csv-output] [--images-path images-path] [--ground-truths-path ground-truths-path]
###### Parameters:

    --mode mode                              |  name of environment
    --csv-input csv-input                    |  number of training episodes
    --csv-output csv-output                  |  application mode (train, test, render)
    --images-path images-path                |  type of algorithm (DQN, DQN+TN, DDQN)
    --ground-truths-path ground-truths-path  |  type of experience replay (basic, prioritized)

**Data generator**
##### python3 src/batch.py [--batches batches]
###### Parameters:

    --batches batches  |  name of file containing output from training

**Classifier training**
##### python3 src/classification.py [--trn-data-size trn-data-size] [--model-name model-name]
###### Parameters:

    --trn-data-size trn-data-size  |  name of file containing output from training
    --model-name model-name        |  name of output file containing graph
    
****
<img height="280" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/meme.jpg">

###### Created by: Petr Buchal, Martin Ivančo, Vladimír Jeřábek
