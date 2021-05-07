# Map-Matching



## 数据准备

### 1、路网数据

##### 最终代码要求格式：.shp

##### 格式转换流程：

1. http://download.geofabrik.de/下载对应大洲的osm.pbf文件
2. https://github.com/Proudc/osm2rn进行Interest region抽取和osm2rn转换

##### 路网数据shp2csv（将shp格式文件进行解析，得到点和边的id及coordinates，便于可视化及轨迹点采样）：

https://github.com/Proudc/shp2csv（部分代码运行时视情况修改）

### 2、轨迹数据

##### 要求格式：t-drive数据格式，其中单个record格式为：

```txt
“trajID,yyyy-MM-dd HH:mm:ss,longitude,latitude”
```

##### 例如：

```txt
1,2015-04-01 00:00:00,103.87326,1.35535
```



## Matching流程

1. https://github.com/Proudc/tptk下载代码

2. 数据清洗，目的是将单条轨迹进行连续轨迹段切分

   ```python
   python main.py --phase clean --tdrive_root_dir ./data/taxi_log_2008_by_id/ --clean_traj_dir ./data/tdrive_clean/
   ```

3. Map-matching

   ```python
   python main.py --phase mm --clean_traj_dir ./data/tdrive_clean/ --rn_path ./data/Beijing-16X16-latest/ --mm_traj_dir ./data/tdrive_mm/
   ```

**Attention：**matching时速度过慢是由于当使用的路网过密时，计算量太大造成的，解决方法是修改networkx包中的a_star算法，添加一个寻找路径时的循环时间控制变量，可以大幅度提升时间。



## 可视化



## 轨迹段连接与采样



附处理singapore_drive_data的流程

1、使用java/util/rawTrajProcess/RawTrajProcess.java中的trajDistinction函数进行轨迹数据按天区分，得到30天的数据，存储于

2、使用java/util/changeNodeAndEdge/ChangeRecordFormat.java中的changeRecordFormat函数进行记录格式转换，将原始数据格式转换为t-drive格式

3、数据清洗

进入/home/changzhihao/miniconda3/envs/tptk/tptk-master目录下执行

```
conda activate tptk
```

之后执行

```
python main.py --phase clean --tdrive_root_dir /mnt/data8/changzhihao/singapore_drive_data/tdrive_format/2015-04-02/ --clean_traj_dir /mnt/data8/changzhihao/singapore_drive_data/tdrive_format/2015-04-02-clean
```

3、使用conf/copy.sh将/mnt/data8/changzhihao/singapore_drive_data/tdrive_format/2015-04-01-clean中的数据复制成好几个文件夹，方便进行并行matching处理

4、对多个文件夹使用python main.py --phase mm --clean_traj_dir ./data/tdrive_clean/ --rn_path ./data/Beijing-16X16-latest/ --mm_traj_dir ./data/tdrive_mm/ 进行matching处理

5、进行轨迹数据连接处理

6、进行轨迹采样