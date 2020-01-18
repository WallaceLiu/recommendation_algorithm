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
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.tree.model.{ GradientBoostedTreesModel, Node, RandomForestModel }
import org.apache.spark.mllib.tree.configuration.Algo.{ Algo, Regression }
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import java.io.{ ObjectInputStream, ObjectOutputStream }
import java.net.URI
import java.sql.Connection
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }

object EnsembleTree {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().
      master("local").
      appName("EnsembleTree").
      getOrCreate()

    import spark.implicits._

    // 1.1 初始化参数
    val dataPath = "hdfs://1.1.1:9000/data/Outbrain/all.csv"
    val minFeature = 10
    val defaultValue = 0.0
    val modelSavePath = ""
    var iteratTree = 10
    var iteratDepth = 10
    var maxAuc = 0.0
    var maxDepth = 10
    var numTrees = 10
    var minInstancesPerNode = 2
    var iter = 100
    var reg_param = 0.0
    var elastic_net_param = 0.0

    // 2.1 取样本数据
    val dataRead = spark.read.options(Map(("delimiter", "|"), ("header", "false"))).csv(dataPath)
    val col = dataRead.columns
    val readSampleData = dataRead.withColumnRenamed(col(0), "label").
      withColumnRenamed(col(1), "feature").
      withColumnRenamed(col(2), "item")
    readSampleData.cache()

    //2.2  建立数据处理方法
    val dataProcessObj1 = new DataProcess()
    val dataProcessObj2 = new DataProcess()
    val dataProcessObjAll = new DataProcess()

    //2 训练样本准备，准备2份
    val (training1, test1) = sampleDataProcess(spark, readSampleData, dataProcessObj1)
    training1.cache()
    training1.count()
    test1.cache()
    test1.count()

    val (training2, test2) = sampleDataProcess(spark, readSampleData, dataProcessObj2)
    training2.cache()
    training2.count()
    test2.cache()
    test2.count()

    //3.1 Gbdt1模型训练
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    boostingStrategy.treeStrategy.minInstancesPerNode = minInstancesPerNode
    boostingStrategy.numIterations = numTrees
    boostingStrategy.treeStrategy.maxDepth = maxDepth
    val gbdtMode1 = GradientBoostedTrees.train(training1.rdd, boostingStrategy)

    //3.2 Gbdt2模型训练
    val gbdtMode2 = GradientBoostedTrees.train(training2.rdd, boostingStrategy)


    //4 解析样本，通过2个树模型映射到最终的LR输入向量
    val gbdtMode1_BC = spark.sparkContext.broadcast(gbdtMode1)
    val gbdtMode2_BC = spark.sparkContext.broadcast(gbdtMode2)

    val mergeSampleData = readSampleData.map { row =>
      val click = row(0).toString().toInt
      val detail = row(1).toString()
      val itemid = row(2).toString()
      val label = if (click > 0) 1.0 else 0.0

      //第1个GBDT映射
      val (tree1Size, tree1NodeFeature) = gettreeNode(gbdtMode1_BC.value, tree1Feature)

      //第2个GBDT映射
      val (tree2Size, tree2NodeFeature) = gettreeNode(gbdtMode2_BC.value, tree2Feature)

      //所有样本归一化
      val allFeature = allMap
      val allSize = dataProcessObjAll.numFeatures

      //合并
      val mergeFeature = (tree1NodeFeature ++
        (tree2NodeFeature.map(f => (f._1 + tree1Size.toInt, f._2))) ++
        (tree3NodeFeature.map(f => (f._1 + tree1Size.toInt + tree2Size.toInt, f._2))) ++
        (allFeature.map(f => (f._1 + tree1Size.toInt + tree2Size.toInt + tree3Size.toInt, f._2)))).sortBy(f => f._1)
      val mergeSize = tree1Size + tree2Size + tree3Size + allSize
      val point = LabeledPoint(label, Vectors.sparse(mergeSize.toInt, mergeFeature.map(_._1), mergeFeature.map(_._2)))
      point
    }

    //5 lr模型训练
    val Splits = mergeSampleData.randomSplit(Array(0.7, 0.3))
    val Training = Splits(0)
    val Test = Splits(1)
    Training.cache()
    Test.cache()
    Training.count()
    Test.count()

    val lr = new LogisticRegressionWithLBFGS().setNumClasses(2)
    lr.optimizer.setNumIterations(iter)
    lr.optimizer.setRegParam(reg_param)
    val lrModel = lr.run(Training.rdd)

    //6 计算模型指标
    lrModel.clearThreshold()
    val scoreAndLabels = Test.rdd.map { point =>
      val score = lrModel.predict(point.features)
      (score, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auc = metrics.areaUnderROC()
    val aupr = metrics.areaUnderPR()
    println(s"AUC: ${auc}")
    println(s"AUPR: ${aupr}")

    // 7.1 封装模型
    val mllibEST = new EnsembleTreeModel()
    // 7.2 保存模型
    modelSave(mllibEST, modelSavePath)

  }

}

