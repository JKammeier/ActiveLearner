# ActiveLearner
 An active learner for a student project.

[ActiveLearner.py](./ActiveLearner.py) contains the class activeLearner. To train it create an instance and use the class method "activeLearningLoop()".

[ActiveLearner_alternativeTraining.py](./ActiveLearner_alternativeTraining.py) is the same as [ActiveLearner.py](./ActiveLearner.py), but instead of keeping the trained model from the previous iteration of the active learning loop and continuing to train that, a new model gets build in every iteration.

[AL_Tester.py](./AL_Tester.py) is used to to run tests on this model to evaluate its performance.

[image_creator.py](./image_creator.py) is a tool to create images of the three most and least informative instances of the dataset.

[tf_env.yaml](./tf_env.yaml) contains an Anaconda environment with all necessary dependancies for a windows machine. The necessary Python packages are: Tensorflow 2.10, matplotlib

