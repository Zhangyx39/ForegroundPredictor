import org.apache.spark.sql.SparkSession

/**
  * Created by Yixing Zhang on 11/30/17.
  */
object Test {
    def main(args: Array[String]) {
        val spark = SparkSession
            .builder()
            .appName("Foreground Predictor")
            .master("local[*]")
            .getOrCreate()

        val file = spark.read.csv("input")

        println(file.filter(row => row.getString(3087).equals("0")).count())
    }
}
