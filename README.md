A Bayesian Neural Network, that can decide whether a skin lession is a malign melanoma or not.

You need CUDA 10.0 and CUDNN 7.6 in order to run this code in gpu-mode! If you haven't you can instead use the requirements_cpu.txt to install the libraries, but training might be very slow then!

Run "sh setup.sh" to setup the repository, install the python environment and download the data.

Execute "python main.py" in order to run the program. You will get into the interactive mode in order to give you a better control what you are doing. One example you could run in the interactive mode is:

model = model_factory.create_model(config=config)

test_images, test_labels = dataloader.load_and_preprocess_multiclass_validation_data(config=config)

model_factory.train_model(model, test_images, test_labels, config=config)

The two main objectives of this program are 1) to increase the f1_score of the validation data and 2) to increase the proportion_of_mistakes_in_top_10 in order to validate the correlation of the error with the uncertainties. 

Features like multi-class classification of different kinds of scin cancer are work in progress.