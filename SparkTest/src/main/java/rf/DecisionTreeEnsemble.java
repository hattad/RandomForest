package rf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import scala.Tuple2;

public class DecisionTreeEnsemble{
	//private static int LIMIT = 0;
	private static int NUM_TREES = 0;
	private static int NUM_ATTRIBUTES = 0;
	private static Set<Double> CLASSLABEL_VALUES = new HashSet<Double>();
	
	public static void main(String args[]){
	SparkConf sparkConf = new SparkConf().setAppName("DecisionTreeEnsemble");
	JavaSparkContext sc = new JavaSparkContext(sparkConf);
	Map<String, String> treeConfig = RandomForestHelper.getServerMap();
	
	// Load and parse the data file.
	// Cache the data since we will use it again to compute training error.
	String path = args[0];
	final int classLabelIndex = Integer.parseInt(treeConfig.get(Constants.CLASS_LABEL_INDEX));
	JavaRDD<String> data = sc.textFile(path);
	JavaRDD<LabeledPoint> parsedData = data.map(
			new Function<String, LabeledPoint>() {
	    public LabeledPoint call(String line) {
	      String[] parts = line.split(",");
	      int vectorSize = parts.length - 1;
	      Double classLabel = null;
	      double[] v = new double[vectorSize];
	      for (int i = 0; i < vectorSize; i++)
	            v[i] = Double.parseDouble(parts[i]);
	      classLabel = Double.parseDouble(parts[classLabelIndex]);
	      CLASSLABEL_VALUES.add(classLabel);
	      return new LabeledPoint(classLabel, Vectors.dense(v));
	    }
	  }
	);
	//60% of data for training and other 40 for test
	JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[]{0.6, 0.4});
	JavaRDD<LabeledPoint> trainData = splits[0];
	JavaRDD<LabeledPoint> testData = splits[1];
    trainData.cache();
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
	
	// Set parameters.
	Integer numClasses = CLASSLABEL_VALUES.size();
	
	// Empty categoricalFeaturesInfo indicates all features are continuous.	
	HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
	String impurity = "gini";
	Integer maxDepth = 5;
	Integer maxBins = 100;
	
	final List<HashSet<Integer>> featureIndex = new ArrayList<HashSet<Integer>>();
	Random random = new Random(); 
	if(0 != numFeatures){
		for(int i = 0; i < NUM_TREES; i++){
			HashSet<Integer> set = new HashSet<Integer>();
			while(set.size() < NUM_ATTRIBUTES) {
				random.setSeed(System.currentTimeMillis());
				set.add((int) (random.nextInt(NUM_ATTRIBUTES)));
			}
			featureIndex.add(set);
		}
	}
	
	List<JavaRDD<LabeledPoint>> samples = new ArrayList<JavaRDD<LabeledPoint>>();
	for(int i = 0 ; i < NUM_TREES ; i++){
		//LIMIT = i * NUM_TREES;
		final Set<Integer> set = featureIndex.get(i);
		samples.add(trainData.map(new Function<LabeledPoint, LabeledPoint>() {
			public LabeledPoint call(LabeledPoint p) throws Exception {
		    	double[] featureList = p.features().toArray();
		    	double[] v = new double[NUM_ATTRIBUTES];
		    	int j =0;
		    	for(int i : set){
		        	v[j] = featureList[i];
		        	j++;
		        } 
		        return new LabeledPoint(p.label(), Vectors.dense(v));
		    }
		}));
	}
	//String sampleDatapath1 = "examples/src/main/resources/sampletree2";
	//MLUtils.saveAsLibSVMFile(sample2.rdd(), sampleDatapath1);
	// Train a DecisionTree model for classification.
	JavaPairRDD<Double, Double> predictionAndLabel = null;
	for(int i = 0 ; i < NUM_TREES ; i++){
		final DecisionTreeModel model = DecisionTree.trainClassifier(samples.get(i), numClasses,
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
	
	Double trainErr =
	1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
	  public Boolean call(Tuple2<Double, Double> pl) {
	    return !pl._1().equals(pl._2());
	  }
	}).count() / (testData.count() * NUM_TREES);
	System.out.println("Training error: " + trainErr);
	}
}
