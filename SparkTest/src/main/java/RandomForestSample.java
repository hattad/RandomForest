import java.util.HashMap;

import javax.xml.bind.JAXBException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

public class RandomForestSample {
	  public static void main(String[] args) {
		final long startTime = System.currentTimeMillis();
		SparkConf sparkConf = new SparkConf().setAppName("JavaRandomForestClassification");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		// Split the data into training and test sets (30% held out for testing)
		String path = args[0];
		final int classLabelIndex = 4;
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
		      return new LabeledPoint(classLabel, Vectors.dense(v));
		    }
		  }
		);
		JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[]{0.6, 0.4});
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];
	
		// Train a RandomForest model.
		//  Empty categoricalFeaturesInfo indicates all features are continuous.
		Integer numClasses = 3;
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		Integer numTrees = 3; // Use more in practice.
		String featureSubsetStrategy = "sqrt"; // Let the algorithm choose.
		String impurity = "gini";
		Integer maxDepth = 5;
		Integer maxBins = 100;
		Integer seed = 12345;
	
		final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
		  categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
		  seed);
		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> predictionAndLabel =
		  testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
		    public Tuple2<Double, Double> call(LabeledPoint p) {
		      return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
		    }
		  });
		Double testErr =
		  1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
		    public Boolean call(Tuple2<Double, Double> pl) {
		      return !pl._1().equals(pl._2());
		    }
		  }).count() / testData.count();
		System.out.println("Test Error: " + testErr);
		System.out.println("Learned classification forest model:\n" + model.toDebugString());
		final long endTime = System.currentTimeMillis();
		System.out.println("Total execution time: " + (endTime - startTime) );
		try {
			RandomForestToPMML.createObject(model);
		} catch (JAXBException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	  }
}
