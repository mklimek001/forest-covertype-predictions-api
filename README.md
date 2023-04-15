# OpenX recruitment task
Predicting forest cover type from cartographic data

**simple_models.py** file contains a heuristic function which uses connection between tree type and elevation.  

**model_comparison.py** was used to evaluate prepared models and plot graphs  

**api.py** - simple rest api which allows to choose model and predict cover type class

## Data
In folder ‘data’ are original files with data set downloaded from [UCI Machine Learning Repository: Covertype Data Set](https://archive.ics.uci.edu/ml/datasets/Covertype). There are also prepared data sets for models training, testing and validation.

## Preparation
Folder ‘preparation’ contains code used to prepare data and models:
- **data_preparation.py** - reading original dataset, selecting parts for training, testing and validation, preparing scaler model
- **sklearn_models_preparation.py** - training k neighbors classifier and random forest classifier from scikit-learn, grid searching to find best hyperparameters
- **best_params_for_nn_preparation.py** - creating neural network wich 12 Dense layers and Sparse Categorical Cross Entropy loss function, grid search with cross validation to find best hyperparameters (learning rate, number of epochs and batch size)  for neural network
- **nn_model_preparation.py** - training Tensorflow neural network model with best params

![nn_curves](https://user-images.githubusercontent.com/74615934/232141354-6cfc708a-0a43-4a1f-953d-a429ebc92d62.png)

## Models
Folder ‘models’ contain prepared models. 


## Conclusions
![accuracy](https://raw.githubusercontent.com/mklimek001/openx-recruitment-task/main/plots/classification_accuracy.png)
Machine learning models are definitely better than simple heuristic. Their accuracy of each model is very similar. 
- K nearest neighbors classifier - 72,43%
- Random forest classifier - 73,21%
- Simple neural network - 74,54 %

Thanks to confusion matrices we know, that every model has a huge problem with proper detection of class 1 (Spruce/Fir). It is almost always classified as 2 (Lodgepole Pine). Also class 3 is often classified as 6, 7 as 1, 5 as 2 and 5 as 2. 
![confusion_matrix](https://raw.githubusercontent.com/mklimek001/openx-recruitment-task/main/plots/heuristic_confusion_martix.png)
![confusion_matrix](https://raw.githubusercontent.com/mklimek001/openx-recruitment-task/main/plots/knn_confusion_matrix.png)
![confusion_matrix](https://raw.githubusercontent.com/mklimek001/openx-recruitment-task/main/plots/random_forest_confusion_matrix.png)
![confusion_matrix](https://raw.githubusercontent.com/mklimek001/openx-recruitment-task/main/plots/nn_confusion_martix.png)
