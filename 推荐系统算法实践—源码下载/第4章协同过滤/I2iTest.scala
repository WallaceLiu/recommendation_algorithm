package book_code

import scala.math._
import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import scala.collection.mutable.WrappedArray
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

object I2iTest {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("I2iTest")
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._

    /**
     * *********************************
     * 1 数据准备
     * 数据来源：
     * MovieLens 【数据地址：https://grouplens.org/datasets/movielens/】（1M、10M、20M 共三个数据集）
     * *********************************
     */

    // 1.1读取item配置表
    val item_conf_path = "hdfs://1.1.1.1:9000/I2I/movies.csv"
    val item_conf_df = spark.read.options(Map(("delimiter", ","), ("header", "true"))).csv(item_conf_path)
    val item_id2title_map = item_conf_df.select("movieId", "title").collect().map(row => (row(0).toString(), row(1).toString())).toMap
    val item_id2genres_map = item_conf_df.select("movieId", "genres").collect().map(row => (row(0).toString(), row(1).toString())).toMap

    // 1.2读取用户行为数据
    val user_rating_path = "hdfs://1.1.1.1:9000/I2I/ratings.csv"
    val user_rating_df = spark.read.options(Map(("delimiter", ","), ("header", "true"))).csv(user_rating_path)

    user_rating_df.dtypes
    val user_ds = user_rating_df.map {
      case Row(userId: String, movieId: String, rating: String, timestamp: String) =>
        ItemPref(userId, movieId, rating.toDouble)
    }
    println("user_ds.show(10)")
    user_ds.show(10)
    user_ds.cache()
    user_ds.count()

    /**
     * *********************************
     * 2 相似度计算
     * *********************************
     */
    val item_id2title_map_BC = spark.sparkContext.broadcast(item_id2title_map)
    val item_id2genres_map_BC = spark.sparkContext.broadcast(item_id2genres_map)

    // 2.1 同现相似度
    val items_similar_cooccurrence = ItemSimilarity.CooccurrenceSimilarity(user_ds).map {
      case ItemSimi(itemidI: String, itemidJ: String, similar: Double) =>
        val i_title = item_id2title_map_BC.value.getOrElse(itemidI, "")
        val j_title = item_id2title_map_BC.value.getOrElse(itemidJ, "")
        val i_genres = item_id2genres_map_BC.value.getOrElse(itemidI, "")
        val j_genres = item_id2genres_map_BC.value.getOrElse(itemidJ, "")
        (itemidI, itemidJ, similar, i_title, j_title, i_genres, j_genres)
    }.withColumnRenamed("_1", "itemidI").
      withColumnRenamed("_2", "itemidJ").
      withColumnRenamed("_3", "similar").
      withColumnRenamed("_4", "i_title").
      withColumnRenamed("_5", "j_title").
      withColumnRenamed("_6", "i_genres").
      withColumnRenamed("_7", "j_genres")
    items_similar_cooccurrence.columns
    // 结果打打印
    items_similar_cooccurrence.cache()
    items_similar_cooccurrence.count
    println("items_similar_cooccurrence.show(20)")
    items_similar_cooccurrence.
      orderBy($"itemidI".asc, $"similar".desc).
      select("i_title", "j_title", "i_genres", "j_genres", "similar").
      show(20)

    // 2.2 余弦相似度
    val items_similar_cosine = ItemSimilarity.CosineSimilarity(user_ds).map {
      case ItemSimi(itemidI: String, itemidJ: String, similar: Double) =>
        val i_title = item_id2title_map_BC.value.getOrElse(itemidI, "")
        val j_title = item_id2title_map_BC.value.getOrElse(itemidJ, "")
        val i_genres = item_id2genres_map_BC.value.getOrElse(itemidI, "")
        val j_genres = item_id2genres_map_BC.value.getOrElse(itemidJ, "")
        (itemidI, itemidJ, similar, i_title, j_title, i_genres, j_genres)
    }.withColumnRenamed("_1", "itemidI").
      withColumnRenamed("_2", "itemidJ").
      withColumnRenamed("_3", "similar").
      withColumnRenamed("_4", "i_title").
      withColumnRenamed("_5", "j_title").
      withColumnRenamed("_6", "i_genres").
      withColumnRenamed("_7", "j_genres")
    items_similar_cosine.columns
    // 结果打打印
    items_similar_cosine.cache()
    items_similar_cosine.count
    println("items_similar_cosine.show(20)")
    items_similar_cosine.
      orderBy($"itemidI".asc, $"similar".desc).
      select("i_title", "j_title", "i_genres", "j_genres", "similar").
      show(20)

    // 2.3 欧氏距离相似度
    val items_similar_euclidean = ItemSimilarity.EuclideanDistanceSimilarity(user_ds).map {
      case ItemSimi(itemidI: String, itemidJ: String, similar: Double) =>
        val i_title = item_id2title_map_BC.value.getOrElse(itemidI, "")
        val j_title = item_id2title_map_BC.value.getOrElse(itemidJ, "")
        val i_genres = item_id2genres_map_BC.value.getOrElse(itemidI, "")
        val j_genres = item_id2genres_map_BC.value.getOrElse(itemidJ, "")
        (itemidI, itemidJ, similar, i_title, j_title, i_genres, j_genres)
    }.withColumnRenamed("_1", "itemidI").
      withColumnRenamed("_2", "itemidJ").
      withColumnRenamed("_3", "similar").
      withColumnRenamed("_4", "i_title").
      withColumnRenamed("_5", "j_title").
      withColumnRenamed("_6", "i_genres").
      withColumnRenamed("_7", "j_genres")
    items_similar_euclidean.columns
    // 结果打打印
    items_similar_euclidean.cache()
    items_similar_euclidean.count
    println("items_similar_euclidean.show(20)")
    items_similar_euclidean.
      orderBy($"itemidI".asc, $"similar".desc).
      select("i_title", "j_title", "i_genres", "j_genres", "similar").
      show(20)

    /**
     * *********************************
     * 3 推荐计算
     * *********************************
     */

    // 推荐结果计算
    // 3.1 同现相似度推荐
    val cooccurrence = items_similar_cooccurrence.select("itemidI", "itemidJ", "similar").map {
      case Row(itemidI: String, itemidJ: String, similar: Double) =>
        ItemSimi(itemidI, itemidJ, similar)
    }
    val user_predictr_cooccurrence = ItemSimilarity.Recommend(cooccurrence, user_ds).map {
      case UserRecomm(userid: String, itemid: String, pref: Double) =>
        val title = item_id2title_map_BC.value.getOrElse(itemid, "")
        val genres = item_id2genres_map_BC.value.getOrElse(itemid, "")
        (userid, itemid, title, genres, pref)
    }.withColumnRenamed("_1", "userid").
      withColumnRenamed("_2", "itemid").
      withColumnRenamed("_3", "title").
      withColumnRenamed("_4", "genres").
      withColumnRenamed("_5", "pref")
    user_predictr_cooccurrence.columns
    user_predictr_cooccurrence.cache()
    user_predictr_cooccurrence.count()
    println("user_predictr_cooccurrence.show(20)")
    user_predictr_cooccurrence.orderBy($"userid".asc, $"pref".desc).show(20)

    // 3.2 余弦相似度推荐
    val cosine = items_similar_cosine.select("itemidI", "itemidJ", "similar").map {
      case Row(itemidI: String, itemidJ: String, similar: Double) =>
        ItemSimi(itemidI, itemidJ, similar)
    }
    val user_predictr_cosine = ItemSimilarity.Recommend(cosine, user_ds).map {
      case UserRecomm(userid: String, itemid: String, pref: Double) =>
        val title = item_id2title_map_BC.value.getOrElse(itemid, "")
        val genres = item_id2genres_map_BC.value.getOrElse(itemid, "")
        (userid, itemid, title, genres, pref)
    }.withColumnRenamed("_1", "userid").
      withColumnRenamed("_2", "itemid").
      withColumnRenamed("_3", "title").
      withColumnRenamed("_4", "genres").
      withColumnRenamed("_5", "pref")
    user_predictr_cosine.columns
    user_predictr_cosine.cache()
    user_predictr_cosine.count()
    println("user_predictr_cosine.show(20)")
    user_predictr_cosine.orderBy($"userid".asc, $"pref".desc).show(20)

    // 3.3 欧氏距离相似度推荐
    val euclidean = items_similar_euclidean.select("itemidI", "itemidJ", "similar").map {
      case Row(itemidI: String, itemidJ: String, similar: Double) =>
        ItemSimi(itemidI, itemidJ, similar)
    }
    val user_predictr_euclidean = ItemSimilarity.Recommend(euclidean, user_ds).map {
      case UserRecomm(userid: String, itemid: String, pref: Double) =>
        val title = item_id2title_map_BC.value.getOrElse(itemid, "")
        val genres = item_id2genres_map_BC.value.getOrElse(itemid, "")
        (userid, itemid, title, genres, pref)
    }.withColumnRenamed("_1", "userid").
      withColumnRenamed("_2", "itemid").
      withColumnRenamed("_3", "title").
      withColumnRenamed("_4", "genres").
      withColumnRenamed("_5", "pref")
    user_predictr_euclidean.columns
    user_predictr_euclidean.cache()
    user_predictr_euclidean.count()
    println("user_predictr_euclidean.show(20)")
    user_predictr_euclidean.orderBy($"userid".asc, $"itemid".desc).show(20)

    // 推荐结果保存
    val table_date = 20181025
    val recommend_table = "table_i2i_recommend_result"
    user_predictr_cooccurrence.createOrReplaceTempView("df_to_hive_table")
    val insertSql1 = s"insert overwrite table ${recommend_table} partition(ds=${table_date}) select userid, itemid, pref from df_to_hive_table"
    println(insertSql1)
    //    spark.sql(insertSql1)

  }

}