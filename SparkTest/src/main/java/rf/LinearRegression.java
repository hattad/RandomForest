package rf;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.stat.Statistics;

import scala.Tuple2;

public class LinearRegression {
  public static void main(String[] args) {
    SparkConf conf = new SparkConf().setAppName("Linear Regression Example");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load and parse the data
    //String path = "examples/src/main/resources/lpsa.data";
    String path = args[0];
//"examples/src/main/resources/QUARTER_AGGREGATE_ALL_out.csv";
//"examples/src/main/resources/intermediate_train_data_4.csv";
//"examples/src/main/resources/costprediction-unscaled.csv";
    JavaRDD<String> data = sc.textFile(path);
    
    JavaRDD<LabeledPoint> parsedData = data.map(
      new Function<String, LabeledPoint>() {
        public LabeledPoint call(String line) {
          String[] parts = line.split(",");
          int labelIndex = parts.length - 1;
          double[] v = new double[labelIndex];
          for (int i = 0; i < labelIndex ; i++)
                v[i] = Double.parseDouble(parts[i]);
          return new LabeledPoint(Double.parseDouble(parts[labelIndex]), Vectors.dense(v));
        }
      }
    );
    final StandardScalerModel scaler = new StandardScaler(true, true).fit(parsedData.map(
    	new Function<LabeledPoint, Vector>() {
        public Vector call(LabeledPoint point) {
            return point.features();
          }
        }).rdd());
    JavaRDD<LabeledPoint> scaledData =parsedData.map(
    	new Function<LabeledPoint, LabeledPoint>() {
        public LabeledPoint call(LabeledPoint point) {
            return  new LabeledPoint(point.label(), scaler.transform(Vectors.dense(point.features().toArray())));
          }
        });
JavaRDD<LabeledPoint> training = scaledData.sample(false, 0.9);
    training.cache();
    JavaRDD<LabeledPoint> test = scaledData.subtract(training);
    // Building the model
    int numIterations = 100;
    double stepSize = 0.1;
    LinearRegressionWithSGD algorithm = new LinearRegressionWithSGD();
    algorithm.optimizer().setStepSize(stepSize).setNumIterations(numIterations);
    algorithm.setIntercept(true);
    final LinearRegressionModel model = algorithm.run(JavaRDD.toRDD(training));

    // Evaluate model on training examples and compute training error
    JavaRDD<Tuple2<Double, Double>> valuesAndPreds = test.map(
      new Function<LabeledPoint, Tuple2<Double, Double>>() {
        public Tuple2<Double, Double> call(LabeledPoint point) {
          double prediction = model.predict(point.features());
          System.out.println("Prediction is : "+prediction+"   Label is : "+point.label());
          return new Tuple2<Double, Double>(prediction, point.label());
        }
      }
    );
    Double MSE = new JavaDoubleRDD(valuesAndPreds.map(
      new Function<Tuple2<Double, Double>, Object>() {
        public Object call(Tuple2<Double, Double> pair) {
          return Math.pow(pair._1() - pair._2(), 2.0);
        }
      }
    ).rdd()).mean();
    System.out.println("training Root Mean Squared Error = " + Math.sqrt(MSE));
    Double MAE = new JavaDoubleRDD(valuesAndPreds.map(
      new Function<Tuple2<Double, Double>, Object>() {
        public Object call(Tuple2<Double, Double> pair) {
          return Math.abs(pair._1() - pair._2());
        }
      }
    ).rdd()).mean();
    System.out.println("training Mean Absolute Error = " + MAE);
/*    final Double yMean = new JavaDoubleRDD(valuesAndPreds.map(
      new Function<Tuple2<Double, Double>, Object>() {
        public Object call(Tuple2<Double, Double> pair) {
          return pair._2();
        }
      }
    ).rdd()).mean();
    Double numerator = new JavaDoubleRDD(valuesAndPreds.map(
      new Function<Tuple2<Double, Double>, Object>() {
        public Object call(Tuple2<Double, Double> pair) {
          return Math.pow(pair._1() - pair._2(), 2.0);
        }
      }
    ).rdd()).sum();
    Double denominator = new JavaDoubleRDD(valuesAndPreds.map(
      new Function<Tuple2<Double, Double>, Object>() {
        public Object call(Tuple2<Double, Double> pair) {
          return Math.pow(pair._2() - yMean, 2.0);
        }
      }
    ).rdd()).sum();
    Double Rsquared = 1 - (numerator / denominator);
    System.out.println("R Squared value is = " + Rsquared);*/
    JavaRDD<Object> keys = valuesAndPreds.map(
			new Function<Tuple2<Double, Double>, Object>() {
			      public Object call(Tuple2<Double, Double> pair) {
			        return (Object)pair._1();
			      }
			    }
			  );
	JavaRDD<Object> values = valuesAndPreds.map(
			new Function<Tuple2<Double, Double>, Object>() {
			      public Object call(Tuple2<Double, Double> pair) {
			        return (Object)pair._2();
			      }
			    }
			  );
  Double rsquared = Math.pow(Statistics.corr(keys.rdd(), values.rdd(), "pearson"), 2.0);
  System.out.println("R Squared value is = " + rsquared);
  double[] weights = model.weights().toArray();
  for(double weight : weights){
	  System.out.println("Weight is " + weight);
  }  
  System.out.println("Intercept is " + model.intercept());
  }
}
               