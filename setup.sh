mkdir data
mkdir models
mkdir logs
mkdir visualizations
wget https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_Input.zip
unzip download
rm download
mv ISBI2016_ISIC_Part1_Training_Data data
wget https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv
unzip download
rm download
mv ISBI2016_ISIC_Part1_Training_GroundTruth data