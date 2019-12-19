import csv, shutil, os

csv_reader = csv.reader(open('data/ISIC_2019_Training_GroundTruth.csv', 'r'), delimiter=',')
shutil.rmtree('data/isic2019_multiclass', ignore_errors=True)
os.makedirs('data/isic2019_multiclass')

for it, row in enumerate(csv_reader):
	if it % 50 == 0:
		print(it)
	if row[1] == 'MEL':
		continue
	for i in range(len(row)):
		if row[i] == '1.0':
			if not os.path.exists('data/isic2019_multiclass/' + str(i - 1)):
				os.makedirs('data/isic2019_multiclass/' + str(i - 1))
			shutil.move('data/ISIC_2019_Training_Input/' + row[0] + '.jpg', 'data/isic2019_multiclass/' + str(i - 1) + '/' + row[0] + '.jpg')