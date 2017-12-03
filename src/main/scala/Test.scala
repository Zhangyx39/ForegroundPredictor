import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

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


//        val file = spark.read.option("inferSchema", "true").csv("input/L6_1_965381.csv")
        val file = spark.read
            .option("inferSchema", "true")
//            .csv("input/validating/L6_6_972760.csv")
            .csv("input/sample.csv")
//            .textFile("input/validating/L6_6_972760.csv")

//        file.groupBy("_c3087").count().show()

//        println(file.filter(row => row.getString(3087).equals("1")).count())
//        println(file.rdd.filter(row => row.get(3087) == 1).count())

//        println(file.filter(row => row.substring(row.lastIndexOf(",") + 1)
//            .equals("1")).count())

//        file.sample(false, 0.01)
//            .repartition(1)
//            .write.csv("output")



        val ones = file.select("*").where("_c3087 = 1").limit(10)
        val zeros = file.select("*").where("_c3087 = 0")
        val union = ones.union(zeros)

        println(ones.count())
        println(zeros.count())
        println(union.count())
        println(file.count())

//        file.show(5)
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

L6_6
+------+------+
|_c3087| count|
+------+------+
|     1|  8288|
|     0|964472|
+------+------+

sample
+------+-----+
|_c3087|count|
+------+-----+
|     1|  100|
|     0| 9477|
+------+-----+
 */
