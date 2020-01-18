package book_code

import org.apache.spark.sql.{ SparkSession, _ }
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.feature.Word2Vec
import java.util.Date
import java.text.SimpleDateFormat

object Word2vec {

  /**
   * word2vec实现：
   *
   * 1）读取训练样本
   * 2）w2v模型训练
   * 3）提取词向量，并且计算相似词
   *
   * @author sunbow
   */

  def main(args: Array[String]): Unit = {

    /**
     * #############################################################
     *
     * Step 1：初始化
     *
     * ##############################################################
     */

    val spark = SparkSession
      .builder
      .appName("Word2vec")
      .config("spark.hadoop.validateOutputSpecs", "false")
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._
    val data_path = args(0)
    val conf_path = args(1)
    val defaultFS = args(2)
    val NumIterations = args(3).toInt
    val MaxSentenceLength = args(4).toInt
    val MinCount = args(5).toInt
    val VectorSize = args(6).toInt
    val WindowSize = args(7).toInt
    val simil_size = args(8).toInt

    /**
     * #############################################################
     *
     * Step 2：数据准备
     *
     * ##############################################################
     */
    // 2.1读取item配置表
    val id_conf_df = spark.read.options(Map(("delimiter", "|"), ("header", "false"))).csv(conf_path)
    val id2title_map = id_conf_df.collect().map(row => (row(0).toString(), row(1).toString())).toMap

    // 2.2读取样本数据
    val sequence_sample = spark.read.text(data_path).map {
      case Row(id_list: String) =>
        val seq = id_list.split(" ").toSeq
        seq
    }
    sequence_sample.repartition(500).cache()
    sequence_sample.count()
    println("sequence_sample.show()")
    sequence_sample.show()

    /**
     * #############################################################
     *
     * Step 3：Word2Vec
     *
     * ##############################################################
     */
    // 训练模型
    val word2Vec = new Word2Vec().
      setNumIterations(NumIterations).
      setMaxSentenceLength(MaxSentenceLength).
      setMinCount(MinCount).
      setVectorSize(VectorSize).
      setWindowSize(WindowSize)
    val model = word2Vec.fit(sequence_sample.rdd)

    // 模型保存
    val now = new Date()
    val dateFormat1 = new SimpleDateFormat("yyyyMMddHHmmss")
    val time_stamp = dateFormat1.format(now)
    val model_path = s"${defaultFS}/Word2vec/model/${time_stamp}"
    println(model_path)
    model.save(spark.sparkContext, model_path)

    /**
     * #############################################################
     *
     * Step 4：词向量结果保存
     *
     * ##############################################################
     */
    val modelBC = spark.sparkContext.broadcast(model)
    val id2title_map_BC = spark.sparkContext.broadcast(id2title_map)
    // 词，向量，相似词
    val word2vector_rdd = spark.sparkContext.parallelize(model.getVectors.toSeq).map {
      case (word: String, vec: Array[Float]) =>
        // 根据word查找相似word
        val simil_word = modelBC.value.findSynonyms(word, simil_size)
        val simil_word_str = simil_word.map(f => s"${f._1}:${f._2.formatted("%.4f")}").mkString(",")
        val title = id2title_map_BC.value.getOrElse(word, "")
        val simil_title = simil_word.map(f => id2title_map_BC.value.getOrElse(f._1, "")).mkString(",")
        // 向量
        val vec_str = vec.mkString(",")
        (word, vec_str, simil_word, title, simil_title)
    }

    println("word2vector_rdd.toDF().show(30)")
    word2vector_rdd.toDF().withColumnRenamed("_4", "word").withColumnRenamed("_5", "simil_word").select("word", "simil_word").show(20)

    // 结果保存
    val save_path = s"${defaultFS}/Word2vec/model_result/${time_stamp}"
    word2vector_rdd.map(f => s"${f._1}|${f._2}|${f._3}|${f._4}|${f._5}").saveAsTextFile(save_path)

  }

}