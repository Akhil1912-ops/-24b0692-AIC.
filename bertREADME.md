step by step local execution-
   (python -m venv venv
   On Windows: venv\Scripts\activate)
   run this code and setup a virtual environment
   then install the required libraries and import them 

documentation of analysis and experiments
  first load the data and divide them category and texts and split them into 2 parts 
  traing and validation data sets then convert t
  he whole texts  data into embedings using textdatasets
  next with optuna test with which set of parameters the bert model works well.
  and now with that parameters train the model and save the model

final comments.
  we can use optuna with diffrents and wide range of parameters with gpu's and tpu's wich might give
  the great accuracy and well trained model

Notes on error handling and troubleshooting.
  used google colab but due to training crashes and completely used free credits could not complete training there
  so,used the laptop cpu.if again want to get a perfect model can chage the parameters and train on gpu's





