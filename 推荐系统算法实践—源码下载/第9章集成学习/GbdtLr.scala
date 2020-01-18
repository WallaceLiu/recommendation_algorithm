package book_code

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }
import org.apache.spark.mllib.classification.{ LogisticRegressionModel, LogisticRegressionWithLBFGS }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.{ Vector => mlVector }
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.tree.model.{ GradientBoostedTreesModel, Node }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import scala.collection.mutable.ArrayBuffer

object GbdtLr {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().
      master("local").
      appName("GbdtLr").
      getOrCreate()

    import spark.implicits._

    //1 参数准备
    val iteratTree = 10
    val iteratDepth = 10
    val maxAuc = 0.0
    val maxDepth = 15
    val numTrees = 10
    val minInstancesPerNode = 2

    //2 训练样本准备    
    val dataPath = "hdfs://1.1.1.1:9000/user/data01/"

    //2 训练样本准备
    val (trainingData, testData) = readLibSvmSampleData(spark, dataPath)
    trainingData.cache()
    testData.cache()
    println(s"trainingData.count(): ${trainingData.count()}")
    println(s"testData.count(): ${testData.count()}")
    println("trainingData.show")
    trainingData.show
    val data = trainingData.unionAll(testData)

    //3 Gbdt模型训练
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    boostingStrategy.treeStrategy.minInstancesPerNode = minInstancesPerNode
    boostingStrategy.numIterations = numTrees
    boostingStrategy.treeStrategy.maxDepth = maxDepth
    val gbdtModel = GradientBoostedTrees.train(trainingData.rdd, boostingStrategy)

    //4 gbdt模型解析：取出所有树的叶子节点
    val treeLeafMap = getTreeLeafMap(gbdtModel)

    //5 样本数据转换成gbdt叶子节点编号的样本
    val lrSampleLablePoint = lrSample(data.rdd, treeLeafMap, gbdtModel)
    val lrSplits = lrSampleLablePoint.randomSplit(Array(0.7, 0.3))
    val (lrTrainingData, lrTestData) = (lrSplits(0), lrSplits(1))
    lrTrainingData.cache()
    lrTrainingData.count()
    lrTestData.cache()
    lrTestData.count()

    //6 lr模型训练
    val lr = new LogisticRegressionWithLBFGS().setNumClasses(2)
    lr.optimizer.setNumIterations(100)
    lr.optimizer.setRegParam(0.0)
    val lrModel = lr.run(lrTrainingData)

    //7 计算模型指标
    lrModel.clearThreshold()
    val scoreAndLabels = lrTestData.map { point =>
      val score = lrModel.predict(point.features)
      (score, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auc = metrics.areaUnderROC()
    val aupr = metrics.areaUnderPR()
    println(s"AUC: ${auc}")
    println(s"AUPR: ${aupr}")

  }

  /**
   * 根据gbdt模型生成gbdtlr模型的样本
   */
  def lrSample(): RDD[LabeledPoint] = {
    lrSamplLablePoint
  }

  /**
   * gbdt模型解析叶子节点
   */
  def getTreeLeafMap(gbdtModel: GradientBoostedTreesModel): Map[String, Int] = {
    lrFeatureMap
  }

  /**
   * 读取libSVM格式的文件，生成训练样本和测试样本。
   */
  def readLibSvmSampleData(
    @transient spark: org.apache.spark.sql.SparkSession,
    dataPath: String): (Dataset[LabeledPoint], Dataset[LabeledPoint]) = {
    import spark.implicits._
    // 2.1 读取样本
    // 2.3 划分样本
    (training, test)
  }

}

