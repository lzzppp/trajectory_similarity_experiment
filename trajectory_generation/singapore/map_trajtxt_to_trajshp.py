import pandas as pd
import osr
import shapefile

data = pd.read_csv("D:\Data\TAXI.csv", encoding='utf-8')

ID = data['ID'].unique()

outPath = "./line.shp"
file = shapefile.Writer (outPath)
file.field ('TAXI_ID')

for taxi_id in ID:
	taxi = data[data['ID'] == taxi_id]
	line = []
	for index, row in taxi.iterrows ():
		point = []
		point.append (row['LNG'])
		point.append (row['LAT'])
		line.append (point)
	polyline = []
	polyline.append (line)
	file.line (polyline)
	file.record (taxi_id)

file.close ()

# 定义投影
proj = osr.SpatialReference ()
proj.ImportFromEPSG (4326)  # 4326-GCS_WGS_1984;
wkt = proj.ExportToWkt ()

# 写入投影
f = open (outPath.replace (".shp", ".prj"), 'w')
f.write (wkt)  # 写入投影信息
f.close ()  # 关闭操作流