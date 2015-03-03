package rf;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;


public class RandomForestHelper {

	/**
	 * Read the properties file and create a key value Map
	 * 
	 * @return - Map containing all config details
	 */
	public static Map<String, String> getServerMap(){
		Map<String, String> propertyMap = new HashMap<String, String>();
		try{
			Properties properties = new Properties();
			InputStream in = new FileInputStream("C:\\workspace\\SparkTest\\src\\main\\java\\config.properties");
			properties.load(in);			
			Enumeration<?> e = properties.propertyNames();
			while (e.hasMoreElements()) {
				String name = (String) e.nextElement();
				String value = properties.getProperty(name);
				propertyMap.put(name, value);
			}
		} catch(FileNotFoundException fnfe){
			System.out.println("Properties file not found" + fnfe);
		} catch(IOException ioe){
			System.out.println("IOException while reading the properties file" + ioe);
		}
		return propertyMap;
	}

	public static Integer calculateNumTrees(int numFeatures) {
		// Use elbow method to determine number of trees
		// 3 by default
		return 3;
	}

	public static Integer calculateNumAttributes(int numFeatures) {
		// TODO Auto-generated method stub
		return (int) Math.sqrt(numFeatures);
	}
}
