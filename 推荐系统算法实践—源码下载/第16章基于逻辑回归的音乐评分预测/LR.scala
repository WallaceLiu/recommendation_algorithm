package book_code

import org.apache.spark.mllib.classification.{ LogisticRegressionModel, LogisticRegressionWithLBFGS }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession

import java.io.{ ObjectInputStream, ObjectOutputStream }
import java.net.URI
import java.sql.Connection
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }

object LR {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("LR")
      .config("spark.hadoop.validateOutputSpecs", "false")
      .enableHiveSupport()
      .getOrCreate()
    import spark.implicits._

    // 1.1 初始化参数
    val dataPath = "hdfs://1.1.1.1:9000/LR_Data/sample_original_1/all.csv"
    val minFeature = 10
    val defaultValue = 0.0
    val modelSavePath = ""

    val iter = 100
    val reg_param = 0.0
    val elastic_net_param = 0.0

    // 2.2 取样本数据
    val dataRead = spark.read.options(Map(("delimiter", "|"), ("header", "false"))).csv(dataPath)
    val col = dataRead.columns
    val readSampleData = dataRead.withColumnRenamed(col(0), "label").
      withColumnRenamed(col(1), "feature").
      withColumnRenamed(col(2), "item")
    readSampleData.cache()

    //2.3  建立标签ID的索引以及数据处理方法
    val dataProcessObj = new DataProcess()

    // 2.4 生成样本
    val (training, test) = sampleDataProcess(spark, readSampleData, dataProcessObj)
    training.cache()
    training.count()
    test.cache()
    test.count()

    //3.1 建立逻辑回归模型       
    val lr = new LogisticRegressionWithLBFGS().setNumClasses(2)
    lr.optimizer.setNumIterations(iter)
    lr.optimizer.setRegParam(reg_param)
    val lrModel = lr.run(training.rdd)

    //3.2 计算模型指标
    lrModel.clearThreshold()
    val scoreAndLabels = test.rdd.map { point =>
      val score = lrModel.predict(point.features)
      (score, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auc = metrics.areaUnderROC()
    val aupr = metrics.areaUnderPR()
    println(s"AUC: ${auc}")
    println(s"AUPR: ${aupr}")

    // 4.1 封装模型
    val mllibLR = new LrModel(lrModel, defaultValue, dataProcessObj)
    // 4.2 保存模型
    modelSave(mllibLR, modelSavePath)

  }

  /**
   * 保存序列化的模型
   */
  def modelSave(
    model: LrModel,
    path: String): Unit = {
  }

  def sampleDataProcess(): (Dataset[LabeledPoint], Dataset[LabeledPoint]) = {
    (training, test)
  }

}