package example

import sparkapplication.BaseSparkLocal

import scala.collection.mutable

object Example1 extends BaseSparkLocal {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

//    val l = List(("1", "y,q,s"), ("0", "y,q,"), ("1", ",q,s"), ("0", ",q,"), ("0", ",,"))
//    val dataSet1 = spark.sparkContext.parallelize(l).map{ case (label, features) =>
//      val featuresFormat = features.split(",", -1).zipWithIndex
//        .map{ case (feature, index) => if (feature.nonEmpty){s"${index+1}#$feature"}else{""} }
//        .mkString(",") + ",bias"
//      (label, featuresFormat)
//    }
//    dataSet1.foreach(println)


//    val map = new mutable.HashMap[String, String]()
//    map.put("y", "q")
//    map.put("q", "s")
//    map.foreach(println)
//    println("=====================")
//    val y = map.getOrElse("s", "yqs")
//    println(y)
//    println("=====================")
//    map.foreach(println)

    println(math.signum(10.0))
    println(math.signum(0.0))
    println(math.signum(-10.0))

















  }
}
