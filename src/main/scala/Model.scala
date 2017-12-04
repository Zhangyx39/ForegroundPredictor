import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
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
//                        .textFile(inputPath + "/training")
            .rdd.map(toLabeledPoint)
//            .sample(false, 0.1)
            .persist()



        //        trainingSet.foreach(println)
        //        val model = NaiveBayes.train(trainingSet, lambda = 1.0, modelType = "multinomial")
        //        val model = LinearRegressionWithSGD.train(trainingSet, 10)

        //        val model = new LogisticRegressionWithLBFGS()
        //            .setNumClasses(2)
        //            .run(training)

        val tuning = for (
//            impurity <- Array("entropy", "gini");
            impurity <- Array("gini");
            numTrees <- Range(5, 6, 1);
            maxDepth <- Range(5, 16, 5);
            maxBins <- Range(20, 101, 20)
        ) yield {
//            val model = DecisionTree.trainClassifier(training, 2,
//                Map[Int, Int](), impurity, maxDepth, maxBins)
            val model = RandomForest.trainClassifier(training, 2,
    Map[Int, Int](), numTrees, "auto", impurity, maxDepth, maxBins)

            (model, impurity, maxDepth, maxBins, numTrees)
        }

        val testing = spark.read
//            .textFile(inputPath + "/sample2.csv")
            .textFile(inputPath + "/validating")
            .rdd.map(toLabeledPoint)
            .persist()

        tuning.foreach(x => {
            val (model, impurity, maxDepth, maxBins, numTrees) = x
            val truePositive = spark.sparkContext.longAccumulator
            val falsePositive = spark.sparkContext.longAccumulator
            val falseNegative = spark.sparkContext.longAccumulator
            val trueNegative = spark.sparkContext.longAccumulator

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


            println(impurity + ", " + maxDepth + ", " + maxBins + ", " + numTrees)
            println(truePositive.value + "\t" + falsePositive.value)
            println(falseNegative.value + "\t" + trueNegative.value)
            val sum = truePositive.value + falsePositive.value +
                falseNegative.value + trueNegative.value
            val t = truePositive.value + trueNegative.value
            println("precision: " + t.toDouble / sum.toDouble)
        })


        //        model.save(spark.sparkContext, outputPath)

        spark.stop()
    }
}

// 951487 / 972760 = 97%

// 55 / 8288