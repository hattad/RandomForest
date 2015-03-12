import java.util.HashMap;

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

public class RegressionTree {
  public static void main(String[] args) {
// Load and parse the data file.
// Cache the data since we will use it again to compute training error.
SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
//.set("spark.default.parallelism", "4")
//.set("spark.executor.memory", "1m");
JavaSparkContext sc = new JavaSparkContext(sparkConf);

 String path = args[0];
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<LabeledPoint> parsedData = data.map(
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
JavaRDD<LabeledPoint> training = parsedData.sample(false, 0.6);
    training.cache();
    JavaRDD<LabeledPoint> test = parsedData.subtract(training);

// Set parameters.
//  Empty categoricalFeaturesInfo indicates all features are continuous.
HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
String impurity = "variance";
Integer maxDepth = 5;
Integer maxBins = 100;

// Train a DecisionTree model.
final DecisionTreeModel model = DecisionTree.trainRegressor(training,
  categoricalFeaturesInfo, impurity, maxDepth, maxBins);

// Evaluate model on training instances and compute training error
JavaPairRDD<Double, Double> predictionAndLabel =
    test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
    public Tuple2<Double, Double> call(LabeledPoint p) {
      return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
    }
  });
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
  //System.out.println("Model is = " + model);
}
}

                 