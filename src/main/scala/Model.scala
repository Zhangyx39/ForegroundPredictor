import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.SparkSession

/**
  * Created by Yixing Zhang on 12/1/17.
  */
object Model {
    def main(args: Array[String]) {
        val spark = SparkSession
            .builder()
            .appName("Foreground Predictor")
//            .master("local[*]")
            .getOrCreate()

        val inputPath = args(0)
        val outputPath = args(1)

        val toLabeledPoint = (row: String) => {
            val size = row.length
            val label = (row.charAt(size - 1) - '0').toDouble

            LabeledPoint(label,
                Vectors.dense(row.substring(0, size - 2)
                    .split(",")
                    .map(_.toDouble)))
        }

        val training = spark.read
//            .textFile(inputPath + "/sample.csv")
            .textFile(inputPath + "/training/L6_4_978344.csv")
            .rdd.map(toLabeledPoint)
//            .sample(false, 0.75)
        val testing = spark.read
//            .textFile(inputPath + "/sample2.csv")
            .textFile(inputPath + "/validating/L6_6_972760.csv")
            .rdd.map(toLabeledPoint)


//        trainingSet.foreach(println)
        //        val model = NaiveBayes.train(trainingSet, lambda = 1.0, modelType = "multinomial")
        //        val model = LinearRegressionWithSGD.train(trainingSet, 10)
        val model =
//        DecisionTree.train(training)
        new LogisticRegressionWithLBFGS()
            .setNumClasses(2)
            .run(training)

        val truePositive = spark.sparkContext.longAccumulator
        var falsePositive = spark.sparkContext.longAccumulator
        var falseNegative = spark.sparkContext.longAccumulator
        var trueNegative = spark.sparkContext.longAccumulator

        testing.foreach(lp => {
            val label = lp.label
            val prediction = model.predict(lp.features)
            if (prediction == 1) {
                if (label == 1) {
                    truePositive.add(1)
                } else if (label == 0) {
                    falsePositive.add(1)
                }
            } else if (prediction == 0) {
                if (label == 0) {
                    trueNegative.add(1)
                } else if (label == 1) {
                    falseNegative.add(1)
                }
            }
        })

        println(truePositive.value + "\t" + falsePositive.value)
        println(falseNegative.value + "\t" + trueNegative.value)
        val sum = truePositive.value + falsePositive.value +
            falseNegative.value + trueNegative.value
        val t = truePositive.value + trueNegative.value
        println("precision: " + t.toDouble / sum.toDouble)

//        model.save(spark.sparkContext, outputPath)

        spark.stop()
    }
}

// 951487 / 972760 = 97%

// 55 / 8288