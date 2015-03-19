import java.util.HashMap;

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

public class DecisionTreeSample{
	public static void main(String args[]){
		SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		// Load and parse the data file.
		// Cache the data since we will use it again to compute training error.
		//String datapath = "examples/src/main/resources/sample_libsvm_data.txt";
		//JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD().cache();
		String path = args[0];
		JavaRDD<String> fileData = sc.textFile(path);
		JavaRDD<LabeledPoint> data = fileData.map(
		                new Function<String, LabeledPoint>() {
		    public LabeledPoint call(String line) {
		      String[] parts = line.split(",");
		      int labelIndex = parts.length - 1;
		      double[] v = new double[labelIndex];
		      for (int i = 0; i < labelIndex; i++)
		            v[i] = Double.parseDouble(parts[i]);
		      return new LabeledPoint(Double.parseDouble(parts[labelIndex]), Vectors.dense(v));
		    }
		  }
		);
		//60% of data for training and other 40 for test
		JavaRDD<LabeledPoint> trainData = data.sample(false, 0.6);
		trainData.cache();
		JavaRDD<LabeledPoint> testData = data.subtract(trainData);
		//Set parameters.
		//Empty categoricalFeaturesInfo indicates all features are continuous.
		Integer numClasses = 2;
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		String impurity = "gini";
		Integer maxDepth = 5;
		Integer maxBins = 100;
		
		//Train a DecisionTree model for classification.
		final DecisionTreeModel model = DecisionTree.trainClassifier(trainData, numClasses,
		categoricalFeaturesInfo, impurity, maxDepth, maxBins);
		
		//Evaluate model on training instances and compute training error
		JavaPairRDD<Double, Double> predictionAndLabel =
		testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
		public Tuple2<Double, Double> call(LabeledPoint p) {
		  return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
		}
		});
		Double trainErr =
		1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
		public Boolean call(Tuple2<Double, Double> pl) {
		  return !pl._1().equals(pl._2());
		}
		}).count() / testData.count();
		System.out.println("Training error: " + trainErr);
		System.out.println("Learned classification tree model:\n" + model);
	}
}