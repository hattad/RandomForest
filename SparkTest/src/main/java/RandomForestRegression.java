import java.util.HashMap;
import java.util.Map;

import javax.xml.bind.JAXBException;

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
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

	public class RandomForestRegression {
		public static void main(String args[]){
			SparkConf sparkConf = new SparkConf().setAppName("JavaRandomForest");
			JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
			String path = args[0];
		    JavaRDD<String> data = sc.textFile(path);
		    JavaRDD<LabeledPoint> parsedData = data.map(
		    		new Function<String, LabeledPoint>() {
		        public LabeledPoint call(String line) {
		          String[] parts = line.split(",");
		          int vectorSize = parts.length - 1;
		          double[] v = new double[vectorSize];
		          for (int i = 0; i < vectorSize; i++)
		                v[i] = Double.parseDouble(parts[i]);
		          return new LabeledPoint(Double.parseDouble(parts[vectorSize]), Vectors.dense(v));
		        }
		      }
		    );
			// Split the data into training and test sets (30% held out for testing)
			JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[]{0.6, 0.4});
			JavaRDD<LabeledPoint> trainingData = splits[0];
			JavaRDD<LabeledPoint> testData = splits[1];
		
			// Set parameters.
			//  Empty categoricalFeaturesInfo indicates all features are continuous.
			Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
			int numTrees = 3;
			String impurity = "variance";
			String featureSubsetStrategy = "sqrt";
			Integer maxDepth = 5;
			Integer maxBins = 100;
			int seed = 12345;
		
			// Train a RandomForest model.
			final RandomForestModel model = RandomForest.trainRegressor(trainingData,
					  categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
		
			// Evaluate model on test instances and compute test error
			JavaPairRDD<Double, Double> predictionAndLabel =
			  testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
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
					try {
						RandomForestToPMML.createObject(model);
					} catch (JAXBException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
		}
}
