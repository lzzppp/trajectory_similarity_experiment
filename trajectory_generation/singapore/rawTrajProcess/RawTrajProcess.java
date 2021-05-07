import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.OutputStreamWriter;
import java.io.File;
import java.io.FileOutputStream;

public class RawTrajProcess {
    
    public static void main(String[] args) {
        String filepath = "/mnt/data8/changzhihao/singapore_drive_data/Output.txt";
        String rootpath = "/mnt/data8/changzhihao/singapore_drive_data/trajDistinction/";
        // lookRawFileFormat(filepath);
        trajDistinction(filepath, rootpath);
    }

    public static void lookRawFileFormat(final String filepath) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(filepath)));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] lines = line.split(",");
                if (lines[0].equals("Taxi_000214") && lines[4].equals("2015-04-01")) {
                    System.out.println(lines[0] + "   " + lines[1] + "   " + lines[2] + "   " + lines[3] + "   "
                            + lines[4] + "   " + lines[5] + "   " + lines[6] + "   " + lines[7]);
                }
                
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void trajDistinction(final String filepath, String rootpath) {
        int count = 0;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(filepath)));
            String line;
            while ((line = reader.readLine()) != null) {
                count += 1;
                String[] lines = line.split(",");
                int trajID = Integer.valueOf(lines[0].substring(5, 11));
                String path = rootpath + lines[4] + "/" + String.valueOf(trajID) + ".txt";
                try {
                    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path, true)));
                    writer.write(line + "\n");
                    writer.close();
                } catch(Exception e) {
                    e.printStackTrace();
                }
                System.out.println("The " + count + "th record is done.");
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
