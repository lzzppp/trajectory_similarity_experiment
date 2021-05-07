import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.BufferedWriter;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.*;
import java.text.*;


public class ChangeNodeAndEdge11 {
    
    public static void main(String[] args) {
        // String beforeNodePath = "E:\\Docu\\Science\\mapData\\singapore\\nodeOSM.txt";
        // String afterNodePath = "E:\\Docu\\Science\\mapData\\singapore\\afterNodeOSM.txt";
        // String beforeEdgePath = "E:\\Docu\\Science\\mapData\\singapore\\edgeOSM.txt";
        // String afterEdgePath = "E:\\Docu\\Science\\mapData\\singapore\\afterEdgeOSM.txt";
        // ArrayList<Vertex> vertexList = readVertex(beforeNodePath);
        // changeEdge(beforeEdgePath, afterEdgePath, vertexList);
        // changeNode(afterNodePath, vertexList);
        String rootReadPath = "/mnt/data4/lizepeng/Singapore/data_split_date/2015-04-11/";
        String rootWritePath = "/mnt/data4/lizepeng/Singapore/data_split_date_tdrive/2015-04-11/";
        // changeGPSPoint(rootReadPath, rootWritePath, 26062);
        changeRecordFormat(rootReadPath, rootWritePath, 26062);
        // changeToGPX(rootReadPath, rootWritePath, 26062);
    }

    public static ArrayList<Vertex> readVertex(String path) {
        ArrayList<Vertex> vertexList = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] words = line.split("\t| ");
                Vertex vertex = new Vertex(Integer.valueOf(words[0]), Double.valueOf(words[2]), Double.valueOf(words[1]));
                vertexList.add(vertex);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return vertexList;
    }

    public static void changeEdge(String beforePath, String afterPath, ArrayList<Vertex> vertexList) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(beforePath)));
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(afterPath)));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] words = line.split("\t");
                int edgeId = Integer.valueOf(words[0]);
                int firstVertex = Integer.valueOf(words[1]);
                int secVertex = Integer.valueOf(words[2]);
                writer.write(edgeId + "\t" + firstVertex + "\t" + secVertex + "\t" + 2 + "\t" +
                                vertexList.get(firstVertex).lat + "\t" + vertexList.get(firstVertex).lon + "\t" +
                                vertexList.get(secVertex).lat + "\t" + vertexList.get(secVertex).lon + "\n");
            }
            reader.close();
            writer.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static void changeNode(String afterPath, ArrayList<Vertex> vertexList) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(afterPath)));
            for (int i = 0; i < vertexList.size(); i++) {
                Vertex vertex = vertexList.get(i);
                writer.write(vertex.id + "\t" + vertex.lat + "\t" + vertex.lon + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void changeGPSPoint(String rootReadPath, String rootWritePath, int trajNum) {
        for (int i = 0; i < trajNum; i++) {
            System.out.println("Is processing " + i + "th txt...");
            String readPath = rootReadPath + String.valueOf(i) + ".txt";
            String writePath;
            if (i >= 0 && i < 10) {
                writePath = rootWritePath + "input_0000" + String.valueOf(i) + ".txt";
            } else if (i >= 10 && i < 100) {
                writePath = rootWritePath + "input_000" + String.valueOf(i) + ".txt";
            } else if (i >= 100 && i < 1000) {
                writePath = rootWritePath + "input_00" + String.valueOf(i) + ".txt";
            } else if (i >= 1000 && i < 10000) {
                writePath = rootWritePath + "input_0" + String.valueOf(i) + ".txt";
            } else {
                writePath = rootWritePath + "input_" + String.valueOf(i) + ".txt";
            }
            try {
                BufferedReader reader = new BufferedReader(new FileReader(new File(readPath)));
                BufferedWriter writer = new BufferedWriter(new FileWriter(new File(writePath)));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] words = line.split(",");
                    String date = words[4];
                    String time = words[5];
                    int intTime = convertTime(date, time, "2015-04-01", "00:00:00");
                    double lon = Double.valueOf(words[2]);
                    double lat = Double.valueOf(words[1]);
                    writer.write(intTime + "," + lat + "," + lon + "," + i + "\n");
                }
                reader.close();
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private static int convertTime(String date, String time, String originalDate, String originaTime) {
        String[] dates = date.split("-");
        String[] times = time.split(":");
        String[] originalDates = originalDate.split("-");
        String[] originalTimes = originaTime.split(":");
        Calendar nowTime = Calendar.getInstance();
        Calendar originalTime = Calendar.getInstance();
        nowTime.set(Integer.valueOf(dates[0]), Integer.valueOf(dates[1]), Integer.valueOf(dates[2]),
                Integer.valueOf(times[0]), Integer.valueOf(times[1]), Integer.valueOf(times[2]));
        originalTime.set(Integer.valueOf(originalDates[0]), Integer.valueOf(originalDates[1]),
                Integer.valueOf(originalDates[2]), Integer.valueOf(originalTimes[0]), Integer.valueOf(originalTimes[1]),
                Integer.valueOf(originalTimes[2]));
        long nowSeconds = nowTime.getTimeInMillis() / 1000;
        long originalSeconds = originalTime.getTimeInMillis() / 1000;
        return (int) ((nowSeconds - originalSeconds));
    }

    private static long getMillSecs(String date, String time) {
        String[] dates = date.split("-");
        String[] times = time.split(":");
        Calendar nowTime = Calendar.getInstance();
        Calendar originalTime = Calendar.getInstance();
        nowTime.set(Integer.valueOf(dates[0]), Integer.valueOf(dates[1]), Integer.valueOf(dates[2]),
                Integer.valueOf(times[0]), Integer.valueOf(times[1]), Integer.valueOf(times[2]));
        originalTime.set(1970, 1, 1, 0, 0, 0);
        long nowSeconds = nowTime.getTimeInMillis();
        long originalSeconds = originalTime.getTimeInMillis();
        return nowSeconds - originalSeconds;
    }

    public static void changeRecordFormat(String rootReadPath, String rootWritePath, int trajNum) {
        for (int i = 0; i < trajNum; i++) {
            System.out.println("Is processing " + i + "th txt...");
            String readPath = rootReadPath + String.valueOf(i) + ".txt";
            String writePath;
            if (i >= 0 && i < 10) {
                writePath = rootWritePath + "input_0000" + String.valueOf(i) + ".txt";
            } else if (i >= 10 && i < 100) {
                writePath = rootWritePath + "input_000" + String.valueOf(i) + ".txt";
            } else if (i >= 100 && i < 1000) {
                writePath = rootWritePath + "input_00" + String.valueOf(i) + ".txt";
            } else if (i >= 1000 && i < 10000) {
                writePath = rootWritePath + "input_0" + String.valueOf(i) + ".txt";
            } else {
                writePath = rootWritePath + "input_" + String.valueOf(i) + ".txt";
            }
            try {
                BufferedReader reader = new BufferedReader(new FileReader(new File(readPath)));
                BufferedWriter writer = new BufferedWriter(new FileWriter(new File(writePath)));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] words = line.split(",");
                    String date = words[4];
                    String time = words[5];
                    double lon = Double.valueOf(words[2]);
                    double lat = Double.valueOf(words[1]);
                    writer.write(i + "," + date + " " + time + "," + lon + "," + lat + "\n");
                }
                reader.close();
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static void changeToGPX(String rootReadPath, String rootWritePath, int trajNum) {
        for (int i = 0; i < trajNum; i++) {
            System.out.println("Is processing " + i + "th txt...");
            String readPath = rootReadPath + String.valueOf(i) + ".txt";
            String writePath;
            if (i >= 0 && i < 10) {
                writePath = rootWritePath + "input_0000" + String.valueOf(i) + ".gpx";
            } else if (i >= 10 && i < 100) {
                writePath = rootWritePath + "input_000" + String.valueOf(i) + ".gpx";
            } else if (i >= 100 && i < 1000) {
                writePath = rootWritePath + "input_00" + String.valueOf(i) + ".gpx";
            } else if (i >= 1000 && i < 10000) {
                writePath = rootWritePath + "input_0" + String.valueOf(i) + ".gpx";
            } else {
                writePath = rootWritePath + "input_" + String.valueOf(i) + ".gpx";
            }
            List<Location> locationList = new ArrayList<>();
            try {
                BufferedReader reader = new BufferedReader(new FileReader(new File(readPath)));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] words = line.split(",");
                    String date = words[4];
                    String time = words[5];
                    double lon = Double.valueOf(words[2]);
                    double lat = Double.valueOf(words[1]);
                    long totalSec = getMillSecs(date, time);
                    Location location = new Location(lat, lon, totalSec);
                    locationList.add(location);
                }
                reader.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
            generateGfx(new File(writePath), "Zhihao Chang", locationList);
        }
    }

    private static void generateGfx(File file, String name, List<Location> points) {

        String header = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?><gpx xmlns=\"http://www.topografix.com/GPX/1/1\" creator=\"MapSource 6.15.5\" version=\"1.1\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"  xsi:schemaLocation=\"http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd\"><trk>\n";
        name = "<name>" + name + "</name><trkseg>\n";

        String segments = "";
        DateFormat df = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");
        for (Location location : points) {
            segments += "<trkpt lat=\"" + location.getLatitude() + "\" lon=\"" + location.getLongitude() + "\"><time>"
                    + df.format(new Date(location.getTime())) + "Z</time></trkpt>\n";
        }

        String footer = "</trkseg></trk></gpx>";

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
            writer.append(header);
            writer.append(name);
            writer.append(segments);
            writer.append(footer);
            writer.flush();
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

class Vertex {
    public int id;
    public double lon;
    public double lat;

    public Vertex(int id, double lon, double lat) {
        this.id = id;
        this.lon = lon;
        this.lat = lat;
    }
}

class Location {
    public double lat;
    
    public double lon;
    
    public long time;

    public Location(double lat, double lon, long time) {
        this.lat = lat;
        this.lon = lon;
        this.time = time;
    }

    public double getLatitude() {
        return this.lat;
    }

    public double getLongitude() {
        return this.lon;
    }

    public long getTime() {
        return this.time;
    }

}
