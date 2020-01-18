import org.apache.spark.ml.classification.{ BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel }
import org.apache.spark.ml.evaluation.{ MulticlassClassificationEvaluator, BinaryClassificationEvaluator }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._

/**
* 读取libSVM格式的文件，生成训练样本和测试样本
* 1）读取文件
* 2）生成标签索引
* 3）样本处理
* 4）样本划分
*/
def readLibSvmSampleData(): ={
}

//1 参数准备
val dataPath = "hdfs://192.168.1.100:9000/Recommended_Algorithm_Action/data01/"
val iter = 500
val reg_param = 0.0
val elastic_net_param = 0.0

//2 训练样本准备
val (training, test) = readLibSvmSampleData(spark, dataPath)
training.cache()
test.cache()
println(s"training.count(): ${training.count()}")
println(s"test.count(): ${test.count()}")
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

//6 对模型进行测试
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
// 不同的阈值，计算不同的F1，然后通过最大的F1找出并重设模型的最佳阈值
val fMeasure = binarySummary.fMeasureByThreshold
fMeasure.show
// 获得最大的F1值
val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
// 找出最大F1值对应的阈值（最佳阈值）
val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
// 将模型的Threshold设置为选择出来的最佳分类阈值
lrModel.setThreshold(bestThreshold)


//16 模型保存与加载
// 保存
lrModel.save("hdfs://192.168.1.100:9000/mlv2/lrmodel")
// 加载
val load_lrModel = LogisticRegressionModel.load("hdfs://192.168.1.100:9000/mlv2/lrmodel")
// 加载测试
val load_predict = load_lrModel.transform(test)
    load_predict.select("features", "label", "probability", "prediction").take(5).foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
}
