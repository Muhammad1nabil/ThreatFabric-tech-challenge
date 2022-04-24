<div id="top"></div>

<br />
<div align="center">
  </a>

  <h3 align="center">ThreatFabric Tech Challenge</h3>

  <p align="center">
    This is my submission to ThreatFabric Tech Challenge
    <br />
    <a href="http://www.nabil-tech-challenge.link"><strong>Explore the website »</strong></a>
    <br />
    <br />
  </p>
</div>


<details>
  <summary>Table of Contents</summary>
  <ul>
    <li><h2>About The Task</h2>
      <ul>
        <li><h3>Building Model</h3></li>
        <li><h3>Deploying Model</h3>
        <ul><h3>create Flask RestAPI</h3></ul>
        <ul><h3>create EC2 Instance and setup the environment</h3></ul>
        <ul><h3>create gunicorn WSGI service</h3></ul>
        <ul><h3>install and configure NGINX</h3></ul>
        <ul><h3>create domain using AWS Route 53</h3></ul>
        </li>
      </ul>
    </li>
    <li><h3>Deliverables</h3></li>
    <li><h3>Final Thoughts</h3></li>
    <li><h3>Contact</h3></li>
  </ul>
</details>


# About The Task

The goal of this task is to assess my expertise in building and deploying ML solution through creating a model to predict the users from their keystroke dynamics.

### The Task is divided into two parts:<br>
## 1. <b>Building Model</b><br>
<b>The objective of this part, is building ML models for user recognition based on their keystroke data.</b> keystroke dynamics is a behavioural biometric which utilizes the unique way a person types to verify the identity of an individual. Typing patterns are predominantly extracted from computer keyboards. the patterns used in keystroke dynamics are derived mainly from the two events that make up a keystroke: the Key-Press and Key-Release. The Key-Press event takes place at the initial depression of a key and the Key-Release occurs at the subsequent release of that key.
<br><br>
<b>In this step,</b><br> a dataset of keystroke information of users is given with following information:<br>
• <b>Train_keystroke.csv:</b> in this dataset the keystroke data from 110 users are collected. All users are asked to type a 13-length constant string 8 times and the keystroke data (keypress time and key-release time for each key) are collected. The data set contains 880 rows and 27 columns. The first column indicates UserID, and the rest shows the press and release time for first to 13th character.
<br><br>
<b>building using the following steps:</b>
1. extracted new features from raw data to build the model. In this regard, four features (Hold Time
“HT”, Press-Press time “PPT”, Release-Release Time “RRT”, Release-Press time “RPT”) are introduced and the definition of each of them are described in Jupyter notebook. For each row in Train_keystroke.csv, these features are generated for each two consecutive keys.

2. calculated mean and standard deviation for each feature per row. As a result, 8 features are created (4 mean and 4 standard
deviation) per row.
3. By using 8 generated features and UserID as class, built and trained three ML models<br> 
  • SVM <br>
  • Random Forest<br> 
  • XGBoost<br>
  <br><b>Note</b> 
  No feature selection, cross validation or
  parameters tuning implemented, models just trained on all data based on default parameters of each classifier

4. saved the three trained models using Pickle.

### <b>all the implementation, are detailed and described in the notebook.</b>

the models had low scores, because no feature selection, cross validation nor parameters tuning. but not only that, the data needed to be more preprocessed, as discribed in the notebook. the data needed to handle the inconsistency in the keystoke dynamics <b>, for example look at user 19.</b>

<br><br>

## 2. <b>Deploying Model</b><br>
The objective of this part, is deploying the three ML models from the previous part as a single API.<br><br>
<b>the deployment followed the following steps:</b><br>

## 1. create Flask RestAPI 
  1. first started with creating a github repo to version control my project.
  2. once I created the repo and cloned it to my local PC, the first thing to do is to create a virtual environment <br>
  `python -m venv venv`

  3. activate venv to install the required packages (Pandas - Scikit learn - XGBoost - Flask - Matplotlib) which stored in `requirements.txt`<br>  
  `pip install -r requirements.txt`
  <br><br><b>the previous steps to set up the environment were used in part one while creating the jupyter notebook</b>

  4. create `flask_app.py` to create our single API for user prediction.
``` python
  from flask import Flask, request, jsonify
  import pickle
  import numpy as np

  app = Flask(__name__)
  app.config['DEBUG'] = False

  
  @app.route('/predict', methods=['POST'])
  def predict_user():
    try:
        # reading request json
        data = request.get_json()
        # getting 'Model' from request to select which model gonna be used and load the corresponding pickle model
        model_name = data['Model']
        if model_name.upper() not in ['SVM', 'RF', 'XGB']:
            return jsonify(msg=f'invalid model, {model_name}'), 400

        model = pickle.load(open(f'{model_name.lower()}.model', 'rb'))

        # creating np array with shape (1, 8) aligned correctly like the training model
        record = np.array([data['HT']['Mean'], data['PPT']['Mean'], data['RPT']['Mean'], data['RRT']['Mean'],
                           data['HT']['STD'], data['PPT']['STD'], data['RPT']['STD'], data['RRT']['STD']]).reshape(1, 8)

        prediction = model.predict(record)
        # response payload has the predicted user ID
        return jsonify({'user': int(prediction)})

    except ValueError:
        return jsonify(
            msg=f'invalid data, {data}'), 400
    except Exception as e:
        return jsonify(
            msg=f'unexpected error, {e}'), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0')
```
5. testing the API by running the script using the correct request first, and then try to create a test cases to enhance the api.
``` json
{
  "Model": "RF",
  "HT": {
    "Mean": 48.43,
    "STD": 23.34
  },
  "PPT": {
    "Mean": 120.43,
    "STD": 37.41
  },
  "RRT": {
    "Mean": 124.43,
    "STD": 45.34
  },
  "RPT": {
    "Mean": 132.56,
    "STD": 47.12
  }
}
```
"Model" will be the selected model name and the rest will be the 8 input features.

the response if the data is valid, status code 200:
``` json
{
  "user": "{userID:int}"
}
```

the response if "Model" is invalid, 
status code 400:

``` json
{
  "msg": "invalid model, {request 'Model' value}"
}
```
the response if any value is invalid, status code 400:

``` json
{
  "msg": "invalid data, {request data}"
}
```
the response if unexpected error occur, status code 418:

``` json

{
  "msg": "unexpected error, {exception msg}"
}
```

## 2. create EC2 Instance and setup the environment
logging in AWS platform to create deploy flask app
  1. creating EC2 Instance<br>
      * t2.micro
      * Amazon Linux
      * no extra storage
      * security group {SSH: 22, HTTP: 80, HTTPS: 443}
      
  2. create a new key pair for SSH connection while using PuTTY, and PuTTYgen to generate `.ppk` ssh auth file as AWS will provide `.pem` auth file. -following AWS instructions to use PuTTY-

  3. and from that point, aws terminal is used and ready to clone the repo and setup Flask app - same first 3 steps are repeated here but this time is on EC2 instance-
  <br> while doing these steps, I noticed python version on EC2 instance is not the same from which I used in my local PC. so I installed python 3.8 on EC2 and from that I used it
    * clone git repo
    * `python3.8 -m venv venv` 
    * `source venv/bin/activate`
    * `(venv) pip install -r requirements.txt`
    * running flask again to test everything is working properly
    <br>  `(venv) python3.8 flask_app.py`

## 3. create gunicorn WSGI service
  1. install <b>gunicorn</b> and switch debug bool value to False to be ready for deployment
  2. create `wsgi.py`
  ``` python
  from flask_app import app

  if __name__ == '__main__':
    app.run()
  ```
  3. creating gunicorn service
    * `cd /etc/systemd/system`
    * `sudo nano flask.service`
  ```
    [Unit]
    #  specifies metadata and dependencies
    Description=Gunicorn instance to serve keystroke flask endpoint
    After=network.target

    # tells the init system to only start this after the networking target has been reached

    [Service]
    # We'll map out the working directory and set the PATH environmental variable so that the init system knows where our the executables
    # for the process are located (within our virtual environment).
    
    WorkingDirectory=/home/ec2-user/ThreatFabric-tech-challenge
    Environment=/home/ec2-user/ThreatFabric-tech-challenge/venv/bin

    # We'll then specify the commanded to start the service
    
    ExecStart=/home/ec2-user/ThreatFabric-tech-challenge/venv/bin/gunicorn -b localhost:8000 wsgi:app

    # This will tell systemd what to link this service to if we enable it to start at boot.
    # We want this service to start when the regular multi-user system is up and running:
    
    [Install]
    WantedBy=multi-user.target
  ```
4. `sudo systemctl daemon-reload`
5. `sudo systemctl start flask.service`
6. `sudo systemctl enable flask.service`
7. check `flask.service` is running from `sudo systemctl status flask.service`

## 4. install and configure <b>NGINX</b>

1. install <b>nginx</b><br>
  `sudo yum install nginx`
 <br>got an error to enable from AWS to enable NGINX before installing, so I did that and it worked<br>
 `sudo amazon-linux-extras install nginx1`
2. modified `/etc/nginx/nginx.conf` to add http server configurations and reverse proxy to gunicorn service
  ```
  server {
    server_name www.nabil-tech-challenge.link;
    location / {
      proxy_pass http://localhost:8000;
    }
  }
  ```
  3. `sudo systemctl restart nginx` to read the changes

## 5. create domain using <b>AWS Route 53</b>
whats left is the domain. Using <b>AWS route 53</b> to create `www.nabil-tech-challenge.link` domain and link it to EC2 instance
  
  1. open route 53 console
  2. create new hosted zone if there isn't any to have the domain name
  3. create a record to route traffic to EC2 instance by using the name of the record that has been created in this procedure. 

  4. nginx service checked running, then check it's running publicly
`http://www.nabil-tech-challenge.link`

  5. returned Flask error not found as there's no index route created.
  6. creating index route to state instructions of how to use `/predict` API with an example
  7. pull the changes and restart the services.


<br><br>


## Deliverables
as requested those are my deliverables
1. a complete Jupyter notebook file for part 1 - Building Model<br>
`Part 1 - Building Model.ipynb`
2. component diagram showing modules and services used and their relationships for <b>AWS solution</b> 
`AWS component diagram.jpg` and you can access `http://www.nabil-tech-challenge.link` for more info
3. this `README.md` and jupyter notebook as full-detailed step by step of how I built the model and deployed it

4. also the complete last updated repo compressed
 `nabil-submission.zip`

## Final Thoughts
  I interacted with cloud and deployed web systems and big data systems but this is the first time I actually do it as always there's devops engineer who's responsible to do it. yes I had the knowledge -AWS cloud courses- and now I got the experience. <br>
  for azure approach, I got stuck while deploying the project, and I prefered to focus on AWS -as I know it more and to complete two solutions out of three-. as I had to do the tech challenge while Im working a full-time job and committed to my personal life routine. I didnt have enough time to complete azure solution after I finished AWS deployement.<br><br>

  I really enjoyed the process and devops work, as I finished buidling models and creating flask API in one day, and had to not work for 5 days due to personal reasons. that got me only 4 days to deploy, for me I enjoy my time working with cli/terminals, services/cron-jobs, and especially get stuck in a problem/error which is new to me. error junky I suppose xD

  I wish I fulfilled your requirements and excited to talk to you!

## Contact

Muhammad Nabil - muhammad1nabil@yahoo.com

Project Link: http://www.nabil-tech-challenge.link