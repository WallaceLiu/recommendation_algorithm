package book_code

import org.apache.spark.ml.classification.{ BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel }
import org.apache.spark.ml.evaluation.{ MulticlassClassificationEvaluator, BinaryClassificationEvaluator }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import java.util.Date
import java.text.SimpleDateFormat

object LogisticRegression {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.
      builder().
      appName("LogisticRegression").
      enableHiveSupport().
      getOrCreate()

    import spark.implicits._

    //1 参数准备
    val dataPath = "hdfs://1.1.1.1:9000/user/data01/"
    val iter = 500
    val reg_param = 0.0
    val elastic_net_param = 0.0

    //2 训练样本准备
    val (training, test) = readLibSvmSampleData(spark, dataPath)
    training.cache()
    test.cache()
    println(s"training.count(): ${training.count()}")
    println(s"test.count(): ${test.count()}")
    println("training.show")
    training.show

    //3 建立逻辑回归模型
    val lr = new LogisticRegression().
      setMaxIter(iter).
      setRegParam(reg_param).
      setElasticNetParam(elastic_net_param)

    //4 根据训练样本进行模型训练
    val lrModel = lr.fit(training)

    //5 打印模型信息
    println(s"Coefficients Top 10: ${lrModel.coefficients.toArray.slice(0, 10).mkString(" ")}")
    println(s"Intercept: ${lrModel.intercept}")

    //6 建立多元回归模型
    val mlr = new LogisticRegression().
      setMaxIter(500).
      setRegParam(0.0).
      setElasticNetParam(0.0).
      setFamily("multinomial")

    //7 根据训练样本进行模型训练
    val mlrModel = mlr.fit(training)

    //8 打印模型信息
    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

    //9 对模型进行测试
    val test_predict = lrModel.transform(test)
    test_predict.show
    test_predict.select("features", "label", "probability", "prediction").take(5).foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }

    //10 模型摘要
    val trainingSummary = lrModel.summary

    //11 每次迭代目标值
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    //12 计算模型指标数据
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    //13 模型摘要AUC指标
    val roc = binarySummary.roc
    println("roc.show()")
    roc.show()
    val AUC = binarySummary.areaUnderROC
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    //14 测试集AUC指标
    val evaluator = new BinaryClassificationEvaluator().
      setLabelCol("label").
      setRawPredictionCol("probability").
      setMetricName("areaUnderROC")
    val testAUC = evaluator.evaluate(test_predict)
    println("Test AUC = " + testAUC)

    //15 设置模型阈值
    // 不同的阈值，计算不同的F1，然后通过最大的F1找出并重设模型的最佳阈值。
    val fMeasure = binarySummary.fMeasureByThreshold
    fMeasure.show
    // 获得最大的F1值
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    // 找出最大F1值对应的阈值（最佳阈值）
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
    // 并将模型的Threshold设置为选择出来的最佳分类阈值
    lrModel.setThreshold(bestThreshold)

    //16 模型保存与加载
    // 保存
    val now = new Date()
    val dateFormat1 = new SimpleDateFormat("yyyyMMddHHmmss")
    val time_stamp = dateFormat1.format(now)

    lrModel.save(s"hdfs://1.1.1.1:9000/lrmodel/${time_stamp}")
    // 加载
    val load_lrModel = LogisticRegressionModel.load(s"hdfs://1.1.1.1:9000/lrmodel/${time_stamp}")
    // 加载测试
    val load_predict = load_lrModel.transform(test)
    println("加载测试")
    load_predict.select("features", "label", "probability", "prediction").take(5).foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }

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