
import pickle

if __name__ == "__main__":
	new_data = []
	data = pickle.load(open("shanghai_split_data.pkl", "rb"))
	print(len(data))
	for line in data:
		if len(line) < 30:
			continue
		j = 0
		new_line = []
		while j < len(line):
			if j == len(line) - 1:
				new_line.append(line[j])
				new_data.append(new_line)
				j += 1
				continue
			new_line.append(line[j])
			if len(new_line) == 250:
				new_data.append(new_line)
				new_line = []
			j += 1
	new_new_data = []
	for line in new_data:
		if len(line) < 30:
			continue
		new_new_data.append(line)
	pickle.dump(new_new_data, open("shanghai_taxi_data.pkl", "wb"))
	print(len(new_new_data))