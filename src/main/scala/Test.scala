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

        import spark.implicits._

//        val file = spark.read.option("inferSchema", "true").csv("input/L6_1_965381.csv")
        val file = spark.read
            .option("inferSchema", "true")
            .csv("input/L6_1_965381.csv")

//        file.groupBy("_c3087").count().show()

//        println(file.filter(row => row.getString(3087).equals("1")).count())
//        println(file.rdd.filter(row => row.get(3087) == 1).count())

//        println(file.filter(row => row.substring(row.lastIndexOf(",") + 1)
//            .equals("1")).count())

        file.sample(false, 0.01)
            .repartition(1)
            .write.csv("output")

    }
}

/*
L6_1
+------+------+
|_c3087| count|
+------+------+
|     1| 10202|
|     0|955179|
+------+------+

sample
+------+-----+
|_c3087|count|
+------+-----+
|     1|  100|
|     0| 9477|
+------+-----+
 */
