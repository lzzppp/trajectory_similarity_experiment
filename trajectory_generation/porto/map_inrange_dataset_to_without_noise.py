import pickle
from tqdm import tqdm
from geopy.distance import geodesic

max_speed = 200

def judge_outline(fp, p, bp):
	d1 = geodesic((fp[1], fp[0]), (p[1], p[0])).km
	d2 = geodesic((p[1], p[0]), (bp[1], bp[0])).km
	if (d1/15.0)*3600 > max_speed and (d2/15.0)*3600 > max_speed:
		return True
	return False

if __name__ == "__main__":
	# 采取消除离群点的方法是 将离群点位置点替换为前后两点的空间中点
	porto_without_noise_dataset = []
	porto_noise_dataset = pickle.load(open("porto_in_range_dataset.pkl", "rb"))
	for trajectory in tqdm(porto_noise_dataset): # lon lat
		if len(trajectory) < 10:
			continue
		without_trajectory = [trajectory[0]]
		for i in range(1, len(trajectory) - 1):
			outline_judge = judge_outline(trajectory[i-1], trajectory[i], trajectory[i+1])
			if outline_judge:
				without_trajectory.append([(trajectory[i-1][0]+trajectory[i+1][0])/2.0, (trajectory[i-1][1]+trajectory[i+1][1])/2.0])
			else:
				without_trajectory.append(trajectory[i])
		without_trajectory.append(trajectory[-1])
		porto_without_noise_dataset.append(without_trajectory)
	pickle.dump(porto_without_noise_dataset, open("porto_without_outline_dataset.pkl", "wb"))