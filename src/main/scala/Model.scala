import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

/**
  * Spark program to train the model to predict foreground of an image
  */
object Model {
    def main(args: Array[String]) {
        val spark = SparkSession
            .builder()
            .appName("Foreground Predictor")
//                        .master("local[*]")
            .getOrCreate()

        val inputPath = args(0)
        val outputPath = args(1)

        // Parse a row of csv file to a labeledPoint
        val toLabeledPoint = (row: String) => {
            val size = row.length
            val label = (row.charAt(size - 1) - '0').toDouble

            LabeledPoint(label,
                Vectors.dense(row.substring(0, size - 2)
                    .split(",")
                    .map(_.toDouble)))
        }

                val images = Array("L6_1_965381.csv", "L6_2_982271.csv",
                    "L6_3_978153.csv", "L6_4_978344.csv")
//        val images = Array("sample.csv", "sample2.csv")

        val models = new ArrayBuffer[DecisionTreeModel]()

        // Read the training set as RDD of labeledPoint and cache it.
        val trainingStart = System.currentTimeMillis()
        images.foreach(image => {
            println("training: " + image)
            val start = System.currentTimeMillis()
            val training = spark.read
                .textFile(inputPath + "/training/" + image)
//                .textFile(inputPath + "/" + image)
                .rdd.map(toLabeledPoint)

            val impurity = "gini"
            val maxDepth = 5
            val maxBins = 50

            val model = DecisionTree.trainClassifier(training, 2,
                Map[Int, Int](), impurity, maxDepth, maxBins)

            models.append(model)
            val end = System.currentTimeMillis()
            println("finished training: " + image)
            println("time spent: " + (end - start) / 1000 + " s")
        })
        val trainingEnd = System.currentTimeMillis()
        println("Total training time: " + (trainingEnd - trainingStart) / 1000 + " s")

        val randomForest = new RandomForestModel(Algo.Classification, models.toArray)

        randomForest.trees.foreach(tree => {
            println(tree.depth + ", " + tree.numNodes)
        })

        val validatingStart = System.currentTimeMillis()
        // Read the validating set as RDD of labeledPoint and cache it.
        val validating = spark.read
//            .textFile(inputPath + "/sample2.csv")
            .textFile(inputPath + "/validating")
            .rdd.map(toLabeledPoint)

        val truePositive = spark.sparkContext.longAccumulator
        val falsePositive = spark.sparkContext.longAccumulator
        val falseNegative = spark.sparkContext.longAccumulator
        val trueNegative = spark.sparkContext.longAccumulator

        // use accumulators to count the confusion matrices
        validating.foreach(lp => {
            val label = lp.label
            val prediction = randomForest.predict(lp.features)
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
        val validatingEnd = System.currentTimeMillis()

        // compute precision
        val sum = truePositive.value + falsePositive.value +
            falseNegative.value + trueNegative.value
        val t = truePositive.value + trueNegative.value
        println("precision: " + t.toDouble / sum.toDouble)
        println("Total validating time: " + (validatingEnd - validatingStart) / 1000 + " s")
        // save the model
        randomForest.save(spark.sparkContext, inputPath + "/model2")

        spark.stop()
    }
}