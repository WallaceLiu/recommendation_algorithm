import scala.math._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import scala.collection.mutable.WrappedArray
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.math._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import scala.collection.mutable.WrappedArray
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

    import spark.implicits._
    /**
     * *********************************
     * 1 数据准备
     * 数据来源：
     * MovieLens 【数据地址：https://grouplens.org/datasets/movielens/】（1M、10M、20M 共三个数据集）
     * *********************************
     */
    // 1.1读取item配置表
    val item_conf_path = "hdfs://192.168.1.100:9000/Recommended_Algorithm_Action/I2I/movies.csv"
    val item_conf_df = spark.read.options(Map(("delimiter", ","), ("header", "true"))).csv(item_conf_path)
    item_conf_df.show(5,false)
    val item_id2title_map = item_conf_df.select("movieId", "title").collect().map(row => (row(0).toString(), row(1).toString())).toMap
    val item_id2genres_map = item_conf_df.select("movieId", "genres").collect().map(row => (row(0).toString(), row(1).toString())).toMap

  // 1.2读取用户行为数据
    val user_rating_path = "hdfs://192.168.1.100:9000/user/Recommended_Algorithm_Action/I2I/ratings.csv"
    val user_rating_df = spark.read.options(Map(("delimiter", ","), ("header", "true"))).csv(user_rating_path)
    user_rating_df.dtypes
    val user_ds = user_rating_df.map {
      case Row(userId: String, movieId: String, rating: String, timestamp: String) =>
        ItemPref(userId, movieId, rating.toDouble)
    }
    user_ds.show(5, false)
    user_ds.cache()
    user_ds.count()

    // 1 (用户：物品) => (用户：(物品集合))
    val user_ds1 = user_ds.groupBy("userid").agg(collect_set("itemid")).withColumnRenamed ("collect_set(itemid)", "itemid_set")
    user_ds1.show(2, false)

  // 2 物品:物品，上三角数据
    val user_ds2 = user_ds1.flatMap { row =>
     val itemlist = row.getAs[scala.collection.mutable.WrappedArray[String]](1).toArray. sorted
     val result = new ArrayBuffer[(String, String, Double)]()
      for (i <- 0 to itemlist.length - 2) {
        for (j <- i + 1 to itemlist.length - 1) {
          result += ((itemlist(i), itemlist(j), 1.0))
        }
      }
      result
    }.withColumnRenamed("_1", "itemidI").withColumnRenamed("_2", "itemidJ").withColumnRenamed("_3", "score")
    user_ds2.show(5, false)

  // 3 计算物品与物品，上三角,同现频次
  val user_ds3 = user_ds2.groupBy("itemidI", "itemidJ").agg(sum("score").as("sumIJ"))
  user_ds3.
  show(5, false)

  // 4 计算物品总共出现的频次
  val user_ds0 = user_ds.withColumn("score", lit(1)).groupBy("itemid").agg(sum("score").as("score"))
  user_ds0.show(5, false)

  // 5 计算同现相似度
  val user_ds4 = user_ds3.join(user_ds0.withColumnRenamed("itemid", "itemidJ").withColumnRenamed("score", "sumJ").select("itemidJ", "sumJ"), "itemidJ")
  user_ds4.show(5, false)

  val user_ds5 = user_ds4.join(user_ds0.withColumnRenamed("itemid", "itemidI").withColumnRenamed("score", "sumI").select("itemidI", "sumI"), "itemidI")
  user_ds5.show(5, false)

    // 根据公式N(i)∩N(j)/sqrt(N(i)*N(j)) 计算
    val user_ds6 = user_ds5.withColumn("result", col("sumIJ") / sqrt(col("sumI") * col("sumJ")))
    user_ds6.show(5, false)

// 6 上、下三角合并
    println(s"user_ds6.count(): ${user_ds6.count()}")
    val user_ds8 = user_ds6.select("itemidI", "itemidJ", "result").union(user_ds6.select($"itemidJ".as("itemidI"), $"itemidI".as("itemidJ"), $"result"))
    println(s"user_ds8.count(): ${user_ds8.count()}")
    user_ds8.show(5, false)

// 7 结果返回
    val out = user_ds8.select("itemidI", "itemidJ", "result").map { row =>
      val itemidI = row.getString(0)
      val itemidJ = row.getString(1)
      val similar = row.getDouble(2)
      ItemSimi(itemidI, itemidJ, similar)
    }
    out.show(5, false)

// 结果增加配置信息
    val item_id2title_map_BC = spark.sparkContext.broadcast(item_id2title_map)
    val item_id2genres_map_BC = spark.sparkContext.broadcast(item_id2genres_map)
    
    val items_similar_cooccurrence = out.map {
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
    items_similar_cooccurrence.cache()
    items_similar_cooccurrence.count

// 查询结果信息，查询各种Case
    items_similar_cooccurrence.
      orderBy($"itemidI".asc, $"similar".desc).
      select("i_title", "j_title", "i_genres", "j_genres", "similar").
      show(20)

// 3.1 同现相似度推荐
    val cooccurrence = items_similar_cooccurrence.select("itemidI", "itemidJ", "similar").map {
      case Row(itemidI: String, itemidJ: String, similar: Double) =>
        ItemSimi(itemidI, itemidJ, similar)
    }
    cooccurrence.show(5)

//   1 数据准备 
    val items_similar_ds1 = cooccurrence
    val user_prefer_ds1 = user_ds

  //   2 根据用户的item召回相似物品
    val user_prefer_ds2 = items_similar_ds1.join(user_prefer_ds1, $"itemidI" === $"itemid", "inner")
    user_prefer_ds2.show(5)

    //   3 计算召回的用户物品得分
    val user_prefer_ds3 = user_prefer_ds2.withColumn("score", col("pref") * col("similar")).select("userid", "itemidJ", "score")
    user_prefer_ds3.show(5)

    //   4 得分汇总
    val user_prefer_ds4 = user_prefer_ds3.groupBy("userid", "itemidJ").agg(sum("score").as("score")).withColumnRenamed("itemidJ", "itemid")
    user_prefer_ds4.show(5)

//   5 用户得分排序结果，去除用户已评分物品
    val user_prefer_ds5 = user_prefer_ds4.join(user_prefer_ds1, Seq("userid", "itemid"), "left").where("pref is null")
    user_prefer_ds5.show(5)

    //  6 结果返回
    val out1 = user_prefer_ds5.select("userid", "itemid", "score").map { row =>
      val userid = row.getString(0)
      val itemid = row.getString(1)
      val pref = row.getDouble(2)
      UserRecomm(userid, itemid, pref)
    }

    // 结果增加配置信息
    val user_predictr_cooccurrence = out1.map {
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

    // 查询结果信息，查询各种Case
    user_predictr_cooccurrence.orderBy($"userid".asc, $"pref".desc).show(20)




