def traj2cell_ids(traj_time, cell_seqs):
	traj_grid_seqs = [[], []]
	# cell_seqs = [engrider.to_str_idx(coord) for coord in traj_time[0]]  # lon lat
	target = cell_seqs[0]
	traj_grid_seqs[0].append(target)
	traj_grid_seqs[1].append(traj_time[0])
	index = 1
	while index < len(cell_seqs):
		if cell_seqs[index] == target:
			index += 1
		else:
			traj_grid_seqs[0].append(cell_seqs[index])
			traj_grid_seqs[1].append(traj_time[index])
			target = cell_seqs[index]
			index += 1
	return traj_grid_seqs

c = [5, 5, 5, 7, 8, 8, 9, 7, 8, 8, 5, 5, 5, 6, 7, 7, 5, 5]
t = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print(traj2cell_ids(t, c))