package lr.adadelta

import java.util.Random
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import breeze.linalg.DenseVector
import org.apache.spark.storage.StorageLevel

class Adadelta extends Serializable {

  var rho = 0.9
  var kesi = 1.0
  var lambda1 = 0.0
  var lambda2 = 1.0
  var weightedLoss = 1.0
  var weightedTruncated = 0.0
  var isNoShowLoss = false
  var iterationNum = 5
  var numPartitions = 200

  /**
    * rho
    *
    * */
  def setRho(value:Double):this.type = {
    require(value >= 0.0 && value <= 1.0, s"rho must be not less than 0 and  must be not more than 1, but it is $value")
    rho = value
    this
  }

  /**
    * kesi
    *
    * */
  def setKesi(value:Double):this.type = {
    require(value > 0.0, s"kesi must be more than 0, but it is $value")
    kesi = value
    this
  }

  /**
    * L1正则化
    *
    * */
  def setLambda1(value:Double):this.type = {
    require(value >= 0.0, s"lambda1 must be not less than 0, but it is $value")
    lambda1 = value
    this
  }

  /**
    * L2正则化
    *
    * */
  def setLambda2(value:Double):this.type = {
    require(value >= 0.0, s"lambda2 must be not less than 0, but it is $value")
    lambda2 = value
    this
  }

  /**
    * 加权损失, 可以用正负样本比例
    *
    * */
  def setWeightedLoss(value:Double):this.type = {
    require(value > 0.0, s"weightedLoss must be more than 0, but it is $value")
    weightedLoss = value
    this
  }

  /**
    * 权重截断, 主要是人工截断权重绝对值较小的值
    *
    * */
  def setWeightedTruncated(value:Double):this.type = {
    require(value >= 0.0, s"weightedTruncated must be not less than 0, but it is $value")
    weightedTruncated = value
    this
  }

  /**
    * 是否显示损失, 不建议显示损失值, 这样增加计算量
    *
    * */
  def setIsNoShowLoss(value:Boolean):this.type = {
    isNoShowLoss = value
    this
  }

  /**
    * 迭代次数
    *
    * */
  def setIterationNum(value:Int):this.type = {
    require(value > 0, s"iterationNum must be more than 0, but it is $value")
    iterationNum = value
    this
  }

  /**
    * 分区数量
    *
    * */
  def setNumPartitions(value:Int):this.type = {
    require(value > 0, s"numPartitions must be more than 0, but it is $value")
    numPartitions = value
    this
  }

  /**
    * 训练
    *
    * dataSet: RDD[(Double, String)]    第一个是label, 第二个是用","隔开的多个特征(全是离散特征, 连续特征先分桶再离散化)
    *
    * */
  def fit(dataSet: RDD[(Double, String)]): RDD[(String, String)] = {
    val dataSetFormat = dataSet.map{ case (label, features) =>
      val featuresFormat = features.split(",", -1).zipWithIndex
        .map{ case (feature, index) => if (feature.nonEmpty) {s"${index+1}#$feature"}else{""} }
        .mkString(",") + ",bias"
      (label, featuresFormat)
    }
    val dataSetRepartition = dataSetFormat.map(k => ((new Random).nextInt(numPartitions), k))
      .repartition(numPartitions).map(k => k._2)
    dataSetRepartition.persist(StorageLevel.MEMORY_AND_DISK)

    val sc = dataSetRepartition.sparkContext
    val rhoBroadcast = sc.broadcast(rho)
    val kesiBroadcast = sc.broadcast(kesi)
    val lambda1Broadcast = sc.broadcast(lambda1)
    val lambda2Broadcast = sc.broadcast(lambda2)
    val weightedLossBroadcast = sc.broadcast(weightedLoss)
    val isNoShowLossBroadcast = sc.broadcast(isNoShowLoss)
    val weightedTruncatedBroadcast = sc.broadcast(weightedTruncated)

    // 训练
    var wgxBroadcast = sc.broadcast(new mutable.HashMap[String, Array[Double]]())
    for (iter <- 0 until iterationNum) {
      val trainResult = dataSetRepartition.mapPartitions(dataIterator => adadeltaTrain(dataIterator, wgxBroadcast.value, rhoBroadcast.value,
        kesiBroadcast.value, lambda1Broadcast.value, lambda2Broadcast.value, weightedLossBroadcast.value, isNoShowLossBroadcast.value))
      trainResult.persist(StorageLevel.MEMORY_AND_DISK)
      // 特征权重的聚类
      val trainWeightResult = trainResult.filter(k => !k._1.equals("LossValue"))
        .aggregateByKey((new DenseVector[Double](Array.fill[Double](3)(0.0)), 0L))(
          (vector, array) => (vector._1 + new DenseVector[Double](array._1), vector._2 + array._2),
          (vector1, vector2) => (vector1._1 + vector2._1, vector1._2 + vector2._2) )
        .map{ case (key, (weight, num)) =>
          val weightAverage = 1.0 * weight/num.toDouble        // 如果更新太慢, 可以把1.0改为一个较大的系数
          (key,  weightAverage.toArray)
        }.filter(k => math.abs(k._2.head) > weightedTruncatedBroadcast.value)
      trainWeightResult.persist(StorageLevel.MEMORY_AND_DISK)
      // 计算平均损失
      if(isNoShowLoss){
        val trainLossResult = trainResult.filter(k => k._1.equals("LossValue"))
          .aggregateByKey((new DenseVector[Double](Array.fill[Double](1)(0.0)), 0L))(
            (vector, array) => (vector._1 + new DenseVector[Double](array._1), vector._2 + array._2),
            (vector1, vector2) => (vector1._1 + vector2._1, vector1._2 + vector2._2)
          ).map(k => (k._1, k._2._1/k._2._2.toDouble))
          .map(k => (k._1, k._2.toArray.head)).first()._2
        println(s"====第${iter+1}轮====平均损失:$trainLossResult====")
      }
      val trainWeightResultMap = mutable.HashMap(trainWeightResult.collectAsMap().toList:_*)
      wgxBroadcast = sc.broadcast(trainWeightResultMap)
      trainResult.unpersist()
      trainWeightResult.unpersist()
    }
    dataSetRepartition.unpersist()

    // 特征权重最后结果
    val featuresResultMap = wgxBroadcast.value.map(k => (k._1, k._2.mkString(",")))
    val featuresResult = sc.parallelize(featuresResultMap.toList, numPartitions)
    featuresResult
  }

  /**
    * 每个分区训练
    *
    * */
  def adadeltaTrain(dataIterator: Iterator[(Double, String)],
                    wgxRaw: mutable.HashMap[String, Array[Double]],
                    rho: Double,
                    kesi: Double,
                    lambda1: Double,
                    lambda2: Double,
                    weightedLoss: Double,
                    isNoShowLoss: Boolean): Iterator[(String, (Array[Double], Long))] = {
    // key是特征id, 数组中分别是第t次w, 第t-1次E[g^2], 第t-1次E[x^2]的值, Long类型是次数
    val wgxUpdate = new mutable.HashMap[String, (Array[Double], Long)]()
    var lossSum = 0.0
    var lossNum = 0L
    for (labelFeatures <- dataIterator) {
      val label = labelFeatures._1
      val featuresArray = labelFeatures._2.split(",", -1).filter(_.nonEmpty)

      // 随机过滤出现次数太少的离散特征, 需要用的时候修改
      val featuresChooseArray = featuresArray.filter(feature =>
        if (wgxRaw.contains(feature)) true
        else {
          feature match {
            case _ if feature.startsWith("first#") =>
              if ((new Random).nextDouble <= 0.5) true
              else false
            case _ if feature.startsWith("two#") =>
              if ((new Random).nextDouble <= 0.5) true
              else false
            case _ => true
          }
        }
      )

      // 计算第t次wx+b
      val sumW = featuresChooseArray.par.map{ feature =>
        val params = wgxRaw.getOrElse(feature, Array(0.0, 0.0, 0.0))
        val w = params.head
        w
      }.sum
      val eHx = math.exp(-sumW)
      // 计算第t次prediction
      val prediction = 1.0 / (1.0 + eHx)

      // 计算第t次的损失值
      if (isNoShowLoss) {
        val loss = if (label > 0.0) -weightedLoss * math.log(math.max(prediction, 1E-20)) else -math.log(math.max(1.0 - prediction, 1E-20))
        val lossL1 = lambda1 * featuresChooseArray.par.map{ feature =>
          val params = wgxRaw.getOrElse(feature, Array(0.0, 0.0, 0.0))
          val w = params.head
          math.abs(w)
        }.sum
        val lossL2 = 0.5 * lambda2 * featuresChooseArray.par.map{ feature =>
          val params = wgxRaw.getOrElse(feature, Array(0.0, 0.0, 0.0))
          val w = params.head
          w * w
        }.sum
        lossSum = lossSum + loss + lossL1 + lossL2
        lossNum = lossNum + 1L
      }

      // 根据第t次结果计算第t次梯度
      // gradient = (weightedLoss * y-y+1) * y^ - weightedLoss * y, 把y分成1或者0讨论, 化简后就是下面代码
      val gradient = if (label > 0.0) weightedLoss * (prediction - label) else prediction - label
      for (feature <- featuresChooseArray) {
        // 加入L1和L2正则化后的梯度
        val oldW = wgxRaw.getOrElse(feature, Array(0.0, 0.0, 0.0))(0)    // 第t次w
        val regularizationL1 = lambda1 * math.signum(oldW)
        val regularizationL2 = lambda2 * oldW
        val gradientRegularization = gradient + regularizationL1 + regularizationL2

        // 第t-1次E[g^2]
        val eggOld = wgxRaw.getOrElse(feature, Array(0.0, 0.0, 0.0))(1)
        // 第t次E[g^2]
        val eggNew = rho * eggOld + (1.0 - rho) * gradientRegularization * gradientRegularization
        // 第t次E[g^2]的RMS
        val eggNewRms = math.sqrt(eggNew + kesi)

        // 第t-1次E[x^2]
        val exxOld = wgxRaw.getOrElse(feature, Array(0.0, 0.0, 0.0))(2)
        // 第t-1次E[x^2]的RMS
        val exxOldRms = math.sqrt(exxOld + kesi)

        // 第t次deltaX
        val deltaX = -(exxOldRms/eggNewRms) * gradientRegularization
        // 第t次E[x^2]
        val exxNew = rho * exxOld + (1.0 - rho) * deltaX * deltaX

        // 第t+1次w
        val newW = oldW + deltaX

        // 更新第t+1次w, 第t次E[g^2], 第t次E[x^2]值的sum和次数
        val featureSum = wgxUpdate.getOrElseUpdate(feature, (Array(0.0, 0.0, 0.0), 0L))._1
        featureSum(0) = featureSum(0) + newW
        featureSum(1) = featureSum(1) + eggNew
        featureSum(2) = featureSum(2) + exxNew
        var featureNum = wgxUpdate.getOrElseUpdate(feature, (Array(0.0, 0.0, 0.0), 0L))._2
        featureNum = featureNum + 1L
        wgxUpdate.put(feature, (featureSum, featureNum))
      }
    }
    wgxUpdate.put("LossValue", (Array(lossSum), lossNum))
    wgxUpdate.toIterator
  }

}
