import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.SparkSession

/**
  * Spark program to predict labels for the testing set.
  */
object Predict {
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

        // load model from file
        val model = RandomForestModel.load(spark.sparkContext, inputPath +
            "/model")

        // make prediction for testing set
        val testing = spark.read
            .textFile(inputPath + "/testing")
            //                        .textFile(inputPath + "/testing")
            .rdd.map(toLabeledPoint)
            .persist()
        val prediction = testing.map(lp => {
            model.predict(lp.features).toInt
        })

        // save output
        prediction.repartition(1).saveAsTextFile(outputPath)

    }
}
