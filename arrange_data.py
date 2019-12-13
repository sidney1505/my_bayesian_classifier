import csv, shutil, os

csv_reader = csv.reader(open('data/ISIC_2019_Training_GroundTruth.csv', 'r'), delimiter=',')
num_positive = 0
num_negative = 0
shutil.rmtree('data/positive_samples', ignore_errors=True)
shutil.rmtree('data/negative_samples', ignore_errors=True)
os.makedirs('data/positive_samples')
os.makedirs('data/negative_samples')

for it, row in enumerate(csv_reader):
	if it % 50 == 0:
		print(it)
	if row[1] == 'MEL':
		continue
	elif row[1] == '1.0':
		shutil.move('data/ISIC_2019_Training_Input/' + row[0] + '.jpg', 'data/positive_samples/' + row[0] + '.jpg')
		num_positive += 1
	else:
		shutil.move('data/ISIC_2019_Training_Input/' + row[0] + '.jpg', 'data/negative_samples/' + row[0] + '.jpg')
		num_negative += 1

print('num_positive: ' + str(num_positive) + ' / ' + str(num_positive + num_negative))
print('num_negative: ' + str(num_negative) + ' / ' + str(num_positive + num_negative))