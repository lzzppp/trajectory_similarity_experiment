def get_segments(mask_len, span_len):
	segs = []
	while mask_len >= span_len:
		segs.append (span_len)
		mask_len -= span_len
	if mask_len != 0:
		segs.append (mask_len)
	return segs

print(get_segments(5, 2))