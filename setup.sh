# create the directories used
mkdir data
mkdir models
mkdir logs
mkdir visualizations
# download and prepare the data
wget https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip
unzip ISIC_2019_Training_Input.zip
rm ISIC_2019_Training_Input.zip
mv ISIC_2019_Training_Input data
wget https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv
mv ISBI2016_ISIC_Part1_Training_GroundTruth data
python arrange_data.py
# create and source the python enviroment
virtualenv -p python3 gpu_env
source gpu_env/bin/activate
pip install -r requirements_gpu.txt