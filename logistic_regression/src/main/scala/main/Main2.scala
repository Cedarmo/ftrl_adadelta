package main

import lr.adadelta.Adadelta
import sparkapplication.BaseSparkLocal

object Main2 extends BaseSparkLocal {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    val data = List((1.0, "gds1,group1,brand1,1,1,1"), (0.0, "gds2,group3,brand3,0,0,0"), (1.0, "gds3,group1,brand2,,1,1"),
      (0.0, ",group2,brand1,1,1,0"), (1.0, "gds4,group2,brand4,,,1"), (0.0, ",group1,brand3,1,0,"))
    val dataRDD = spark.sparkContext.parallelize(data, 2)

    val adadelta = new Adadelta()
      .setRho(0.9)
      .setKesi(1.0)
      .setLambda1(0.01)
      .setLambda2(0.01)
      .setWeightedLoss(1.0)
      .setWeightedTruncated(0.2)
      .setIsNoShowLoss(true)
      .setIterationNum(5)
      .setNumPartitions(3)

    val featuresWeightRDD = adadelta.fit(dataRDD)

    featuresWeightRDD.foreach(println)

  }
}