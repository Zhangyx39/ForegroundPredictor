name := "ForegroundPredictor"

version := "0.1"

scalaVersion := "2.11.11"

val sparkVer = "2.2.0"


libraryDependencies ++= {
    Seq (
        "org.apache.spark" %  "spark-core_2.11" % sparkVer,
        "org.apache.spark" %  "spark-sql_2.11" % sparkVer
    )
}
