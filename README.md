# Evaluation

### FA-cup
If you want to evaluate your results on the FA-cup dataset, please visit [this website](http://socialsensor.iti.gr/results/datasets/72-twitter-tdt-dataset) to get the required script to evaluate your results from the authors of the dataset.

### Covid19
To run the evaluation, simply replace our sample ```covid_predicted``` folder with your predicted folder. We have provided you with 30 topics each with 30 keywords in the sample ```covid_predicted``` folder for everyday day.<br>
In ```topic_evaluation.py``` we choose how many of those topics and keywords to use in evaluation. This exists so we can continuously test different values without running the whole algorithm again from scratch to get the required number of topics and keywords.<br>
The ground truths are provided in the ```ground_truth``` folder. they are also available publically on [Kaggle](https://www.kaggle.com/thelonecoder/labelled-1000k-covid19-dataset).
