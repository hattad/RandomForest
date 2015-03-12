import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import scala.Tuple2;

public class RegressionTreeEnsemble {
	private static int LIMIT = 0;
	private static int NUM_TREES = 0;
	private static int NUM_ATTRIBUTES = 0;
	
  public static void main(String[] args) {
	// Load and parse the data file.
	// Cache the data since we will use it again to compute training error.
	SparkConf sparkConf = new SparkConf().setAppName("RegressionTreeEnsemble");
	JavaSparkContext sc = new JavaSparkContext(sparkConf);
	Map<String, String> treeConfig = RandomForestHelper.getServerMap();
	
	String path = args[0];
	final int classLabelIndex = Integer.parseInt(treeConfig.get(Constants.CLASS_LABEL_INDEX));
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<LabeledPoint> parsedData = data.map(
    		new Function<String, LabeledPoint>() {
        public LabeledPoint call(String line) {
          String[] parts = line.split(",");
          int vectorSize = parts.length - 1;
          double[] v = new double[vectorSize];
          for (int i = 0; i < vectorSize; i++)
                v[i] = Double.parseDouble(parts[i]);
          return new LabeledPoint(Double.parseDouble(parts[classLabelIndex]), Vectors.dense(v));
        }
      }
    );
	
  //60% of data for training and other 40 for test
  	JavaRDD<LabeledPoint> trainData = parsedData.sample(false, 0.6);
      trainData.cache();
     	JavaRDD<LabeledPoint> testData = parsedData.subtract(trainData);
  	final int numFeatures = trainData.first().features().size();
  	
  	NUM_TREES = Integer.parseInt(treeConfig.get(Constants.NUM_OF_TREES));
  	NUM_ATTRIBUTES = Integer.parseInt(treeConfig.get(Constants.NUM_OF_ATTRIBUTES_PER_TREE));
  	if(0 == NUM_TREES){
  		//Use elbow method to find the number of trees
  		NUM_TREES = RandomForestHelper.calculateNumTrees(numFeatures);
  	} else if(NUM_TREES < 3){
  		//Set the minimum trees to 3
  		NUM_TREES = 3;
  	}
  	if(0 == NUM_ATTRIBUTES){
  		//Use a formula to calculate the number of attributes per tree
  		NUM_ATTRIBUTES = RandomForestHelper.calculateNumAttributes(numFeatures);
  	}
  	
	//  Empty categoricalFeaturesInfo indicates all features are continuous.
	HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
	String impurity = "variance";
	Integer maxDepth = 5;
	Integer maxBins = 100;
	
	final List<Integer> featureIndex = new ArrayList<Integer>();
	//Generate random numbers of size equal to the number of trees * number of attributes
	Integer featureSize = NUM_TREES * NUM_ATTRIBUTES;
	if(0 != numFeatures){
		for(int i=0; i < featureSize; i++){
			if(featureIndex.size() <= featureSize) {
				featureIndex.add((int) (Math.random()*(numFeatures-1)));
			}
		}
	}
	
	List<JavaRDD<LabeledPoint>> samples = new ArrayList<JavaRDD<LabeledPoint>>();
	for(int i = 0 ; i < NUM_TREES ; i++){
		LIMIT = i * 3;
		samples.add(trainData.map(new Function<LabeledPoint, LabeledPoint>() {
			public LabeledPoint call(LabeledPoint p) throws Exception {
		    	double[] featureList = p.features().toArray();
		    	double[] v = new double[NUM_ATTRIBUTES];
		    	int j =0;
		    	for(int k=LIMIT; k < LIMIT + NUM_ATTRIBUTES; k++){
		        	v[j] = featureList[(Integer)featureIndex.get(k)];
		        	j++;
		        } 
		        return new LabeledPoint(p.label(), Vectors.dense(v));
		    }
		}));
	}
	
	JavaPairRDD<Double, Double> predictionAndLabel = null;
	for(int i = 0 ; i < NUM_TREES ; i++){
		final DecisionTreeModel model = DecisionTree.trainRegressor(parsedData,
				  categoricalFeaturesInfo, impurity, maxDepth, maxBins);
				
		System.out.println(model);
		
		if(i==0){
			predictionAndLabel =  testData.mapToPair(
					new PairFunction<LabeledPoint, Double, Double>() {
			 public Tuple2<Double, Double> call(LabeledPoint p) {
			   return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
			 }
			});
		} else {
			predictionAndLabel.union(testData.mapToPair(
					new PairFunction<LabeledPoint, Double, Double>() {
			 public Tuple2<Double, Double> call(LabeledPoint p) {
			   return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
			 }
			}));
		}	
	} 
	
	Double MSE = new JavaDoubleRDD(predictionAndLabel.map(
	    new Function<Tuple2<Double, Double>, Object>() {
	      public Object call(Tuple2<Double, Double> pair) {
	        return Math.pow(pair._1() - pair._2(), 2.0);
	      }
	    }
	  ).rdd()).mean();
	  System.out.println("training Root Mean Squared Error = " + Math.sqrt(MSE));
	  Double MAE = new JavaDoubleRDD(predictionAndLabel.map(
	    new Function<Tuple2<Double, Double>, Object>() {
	      public Object call(Tuple2<Double, Double> pair) {
	        return Math.abs(pair._1() - pair._2());
	      }
	    }
	  ).rdd()).mean();
	  System.out.println("training Mean Absolute Error = " + MAE);
	  JavaRDD<Object> keys = predictionAndLabel.map(
				new Function<Tuple2<Double, Double>, Object>() {
				      public Object call(Tuple2<Double, Double> pair) {
				        return (Object)pair._1();
				      }
				    }
				  );
		JavaRDD<Object> values = predictionAndLabel.map(
				new Function<Tuple2<Double, Double>, Object>() {
				      public Object call(Tuple2<Double, Double> pair) {
				        return (Object)pair._2();
				      }
				    }
				  );
		Double rsquared = Math.pow(Statistics.corr(keys.rdd(), values.rdd(), "pearson"), 2.0);
		System.out.println("R Squared value is = " + rsquared);
	}
}

                 