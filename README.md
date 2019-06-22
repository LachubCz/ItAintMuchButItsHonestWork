<h1><img height="60" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/UnIT_logo.png">&nbsp;<img height="60" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/logo.png">&nbsp;UnIT extended 2019 ItAintMuchButItsHonestWork</h1>

You can acces UnIT extended 2019 hackaton assignment [here](https://github.com/LachubCz/ItAintMuchButItsHonestWork/blob/master/assignment.pdf). This repository contains source code created by group ItAintMuchButItsHonestWork. Based on tournament metrics we've reached succes rate of 76.89 % in recognizing ellipses and won first place.

### Pipeline

<img height="579" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/pipeline.PNG">

### Data generator

<img height="379" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/data_generator.PNG">

Usage
-----
**Application**
##### python3 main.py -env env -eps eps -mode mode [-alg alg] [-mem mem] [-net net] [-pu pu] [-mdl mdl] [-init] [-frames frames] [-save_f save_f] [-update_f update_f] [-vids] [-mdl_blueprint] [-dont_save]
###### Parameters:

    -env env            |  name of environment
    -eps eps            |  number of training episodes
    -mode mode          |  application mode (train, test, render)
    -alg alg            |  type of algorithm (DQN, DQN+TN, DDQN)
    -mem mem            |  type of experience replay (basic, prioritized)
    -net net            |  neural network architecture (basic, dueling)
    -pu pu              |  processing unit (CPU, GPU)
    -mdl mdl            |  existing model
    -init               |  initialization of experience replay
    -frames frames      |  number of frames which goes to neural network input (1,2,3,4)
    -save_f save_f      |  model saving frequency
    -update_f update_f  |  target network update frequency
    -vids               |  saving video
    -mdl_blueprint      |  saving pdf with blueprint of neural network
    -dont_save          |  disabling of saving any files

###### Examples

    python3 pgunn/main.py -mode train -env CartPole-v0 -eps 5000 -alg DDQN -net dueling -mem prioritized -pu CPU -init -save_f 25
    python3 pgunn/main.py -mode test -env Acrobot-v1 -eps 100 -mdl models/Acrobot-v1.h5
    python3 pgunn/main.py -mode render -env SpaceInvaders-ram-v0 -eps 200 -mdl models/SpaceInvaders-ram-v0.h5 -frames 4 -vids
**Data generator**
##### python3 visualization.py -filename filename -graph_name graph_name -idx_val idx_val [-coordinate_x coordinate_x] [-coordinate_y coordinate_y] [-lines lines lines ...] [-scatter]
###### Parameters:

    -filename filename          |  name of file containing output from training
    -graph_name graph_name      |  name of output file containing graph
    -idx_val idx_val            |  index of column with analyzed data
    -coordinate_x coordinate_x  |  maximal showed x value in graph
    -coordinate_y coordinate_y  |  maximal showed y value in graph
    -lines lines                |  list of y values for reference constant lines 
    -scatter scatter            |  results from every round

###### Examples

    python3 pgunn/visualization.py -filename output.out -graph_name score -idx_val 6
    python3 pgunn/visualization.py -filename log.txt -graph_name results -idx_val 6 -lines 22 30 -scatter
    python3 pgunn/visualization.py -filename log.out -graph_name moves -idx_val 4 -coordinate_x 50
**Classifier training**
##### python3 visualization.py -filename filename -graph_name graph_name -idx_val idx_val [-coordinate_x coordinate_x] [-coordinate_y coordinate_y] [-lines lines lines ...] [-scatter]
###### Parameters:

    -filename filename          |  name of file containing output from training
    -graph_name graph_name      |  name of output file containing graph
    -idx_val idx_val            |  index of column with analyzed data
    -coordinate_x coordinate_x  |  maximal showed x value in graph
    -coordinate_y coordinate_y  |  maximal showed y value in graph
    -lines lines                |  list of y values for reference constant lines 
    -scatter scatter            |  results from every round

###### Examples

    python3 pgunn/visualization.py -filename output.out -graph_name score -idx_val 6
    python3 pgunn/visualization.py -filename log.txt -graph_name results -idx_val 6 -lines 22 30 -scatter
    python3 pgunn/visualization.py -filename log.out -graph_name moves -idx_val 4 -coordinate_x 50
    
****
<img height="280" src="https://raw.githubusercontent.com/LachubCz/ItAintMuchButItsHonestWork/master/images/meme.jpg">

###### Created by: Petr Buchal, Martin Ivančo, Vladimír Jeřábek
