package book_code

import scala.math._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import scala.collection.mutable.WrappedArray
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

object ItemSimilarity extends Serializable {

  import org.apache.spark.sql.functions._

  /**
   * 关联规则计算.
   * 支持度（Support）：在所有项集中{X, Y}出现的可能性，即项集中同时含有X和Y的概率,P(X U Y)/P(I),I是总事务集
   * 置信度（Confidence）:在先决条件X发生的条件下，关联结果Y发生的概率,P(X U Y)/P(X)
   * 提升度（lift）:在含有X的条件下同时含有Y的可能性与没有X这个条件下项集中含有Y的可能性之比,confidence(X => Y)/P(Y)
   * @param user_rdd 用户评分
   * @param RDD[ItemAssociation] 返回物品相似度
   *
   */
  def AssociationRules(user_ds: Dataset[ItemPref]): Dataset[ItemAssociation] = {
    import user_ds.sparkSession.implicits._
    // 1 (用户：物品) => (用户：(物品集合))
    val user_ds1 = user_ds.groupBy("userid").agg(collect_set("itemid")).withColumnRenamed("collect_set(itemid)", "itemid_set")

    // 2 物品:物品，上三角数据
    val user_ds2 = user_ds1.flatMap { row =>
      val itemlist = row.getAs[WrappedArray[String]](1).toArray.sorted
      val result = new ArrayBuffer[(String, String, Double)]()
      for (i <- 0 to itemlist.length - 2) {
        for (j <- i + 1 to itemlist.length - 1) {
          result += ((itemlist(i), itemlist(j), 1.0))
        }
      }
      result
    }.withColumnRenamed("_1", "itemidI").withColumnRenamed("_2", "itemidJ").withColumnRenamed("_3", "score")

    // 3 计算物品与物品，上三角,同现频次
    val user_ds3 = user_ds2.groupBy("itemidI", "itemidJ").agg(sum("score").as("sumIJ"))

    //4 计算物品总共出现的频次
    val user_ds0 = user_ds.withColumn("score", lit(1)).groupBy("itemid").agg(sum("score").as("score"))
    val user_all = user_ds1.count

    //5 计算支持度（Support）
    val user_ds4 = user_ds3.select("itemidI", "itemidJ", "sumIJ").
      union(user_ds3.select($"itemidJ".as("itemidI"), $"itemidI".as("itemidJ"), $"sumIJ")).
      withColumn("support", $"sumIJ" / user_all.toDouble)

    // user_ds4.orderBy($"support".desc).show

    //6 置信度（Confidence）
    val user_ds5 = user_ds4.
      join(user_ds0.withColumnRenamed("itemid", "itemidI").withColumnRenamed("score", "sumI"), "itemidI").
      withColumn("confidence", $"sumIJ" / $"sumI")

    // user_ds5.orderBy($"confidence".desc).show

    //7 提升度（lift）
    val user_ds6 = user_ds5.
      join(user_ds0.withColumnRenamed("itemid", "itemidJ").withColumnRenamed("score", "sumJ"), "itemidJ").
      withColumn("lift", $"confidence" / ($"sumJ" / user_all.toDouble))

    // user_ds6.orderBy($"lift".desc).show

    // 计算同现相似度  
    val user_ds8 = user_ds6.withColumn("similar", col("sumIJ") / sqrt(col("sumI") * col("sumJ")))
    // user_ds8.orderBy($"similar".desc).show

    // 8 结果返回
    val out = user_ds8.select("itemidI", "itemidJ", "support", "confidence", "lift", "similar").map { row =>
      val itemidI = row.getString(0)
      val itemidJ = row.getString(1)
      val support = row.getDouble(2)
      val confidence = row.getDouble(3)
      val lift = row.getDouble(4)
      val similar = row.getDouble(5)
      ItemAssociation(itemidI, itemidJ, support, confidence, lift, similar)
    }
    out
  }

  /**
   * 余弦相似度矩阵计算.
   * T(x,y) = ∑x(i)y(i) / sqrt(∑(x(i)*x(i))) * sqrt(∑(y(i)*y(i)))
   * @param user_rdd 用户评分
   * @param RDD[ItemSimi] 返回物品相似度
   *
   */
  def CosineSimilarity(user_ds: Dataset[ItemPref]): Dataset[ItemSimi] = {
    import user_ds.sparkSession.implicits._

    // 1 数据做准备
    val user_ds1 = user_ds.
      withColumn("iv", concat_ws(":", $"itemid", $"pref")).
      groupBy("userid").agg(collect_set("iv")).
      withColumnRenamed("collect_set(iv)", "itemid_set").
      select("userid", "itemid_set")

    // 2 物品:物品，上三角数据
    val user_ds2 = user_ds1.flatMap { row =>
      val itemlist = row.getAs[scala.collection.mutable.WrappedArray[String]](1).toArray.sorted
      val result = new ArrayBuffer[(String, String, Double, Double)]()
      for (i <- 0 to itemlist.length - 2) {
        for (j <- i + 1 to itemlist.length - 1) {
          result += ((itemlist(i).split(":")(0), itemlist(j).split(":")(0), itemlist(i).split(":")(1).toDouble, itemlist(j).split(":")(1).toDouble))
        }
      }
      result
    }.withColumnRenamed("_1", "itemidI").withColumnRenamed("_2", "itemidJ").withColumnRenamed("_3", "scoreI").withColumnRenamed("_4", "scoreJ")

    // 3 按照公式计算sim
    // x*y = ∑x(i)y(i)
    // |x|^2 = ∑(x(i)*x(i))
    // |y|^2 = ∑(y(i)*y(i))
    // result = x*y / sqrt(|x|^2) * sqrt(|y|^2)
    val user_ds3 = user_ds2.
      withColumn("cnt", lit(1)).
      groupBy("itemidI", "itemidJ").
      agg(sum(($"scoreI" * $"scoreJ")).as("sum_xy"),
        sum(($"scoreI" * $"scoreI")).as("sum_x"),
        sum(($"scoreJ" * $"scoreJ")).as("sum_y")).
        withColumn("result", $"sum_xy" / (sqrt($"sum_x") * sqrt($"sum_y")))

    // 4 上、下三角合并
    val user_ds8 = user_ds3.select("itemidI", "itemidJ", "result").
      union(user_ds3.select($"itemidJ".as("itemidI"), $"itemidI".as("itemidJ"), $"result"))

    // 5 结果返回
    val out = user_ds8.select("itemidI", "itemidJ", "result").map { row =>
      val itemidI = row.getString(0)
      val itemidJ = row.getString(1)
      val similar = row.getDouble(2)
      ItemSimi(itemidI, itemidJ, similar)
    }
    out
  }

  /**
   * 欧氏距离相似度矩阵计算.
   * d(x, y) = sqrt(∑((x(i)-y(i)) * (x(i)-y(i))))
   * sim(x, y) = n / (1 + d(x, y))
   * @param user_rdd 用户评分
   * @param RDD[ItemSimi] 返回物品相似度
   *
   */
  def EuclideanDistanceSimilarity(user_ds: Dataset[ItemPref]): Dataset[ItemSimi] = {
    import user_ds.sparkSession.implicits._

    // 1 数据做准备
    val user_ds1 = user_ds.
      withColumn("iv", concat_ws(":", $"itemid", $"pref")).
      groupBy("userid").agg(collect_set("iv")).
      withColumnRenamed("collect_set(iv)", "itemid_set").
      select("userid", "itemid_set")

    // 2 物品:物品，上三角数据
    val user_ds2 = user_ds1.flatMap { row =>
      val itemlist = row.getAs[scala.collection.mutable.WrappedArray[String]](1).toArray.sorted
      val result = new ArrayBuffer[(String, String, Double, Double)]()
      for (i <- 0 to itemlist.length - 2) {
        for (j <- i + 1 to itemlist.length - 1) {
          result += ((itemlist(i).split(":")(0), itemlist(j).split(":")(0), itemlist(i).split(":")(1).toDouble, itemlist(j).split(":")(1).toDouble))
        }
      }
      result
    }.withColumnRenamed("_1", "itemidI").withColumnRenamed("_2", "itemidJ").withColumnRenamed("_3", "scoreI").withColumnRenamed("_4", "scoreJ")

    // 3 按照公式计算sim
    // dist = sqrt(∑((x(i)-y(i)) * (x(i)-y(i))))
    // cntSum = sum(1)
    // result = cntSum / (1 + dist)
    val user_ds3 = user_ds2.
      withColumn("cnt", lit(1)).
      groupBy("itemidI", "itemidJ").
      agg(sqrt(sum(($"scoreI" - $"scoreJ") * ($"scoreI" - $"scoreJ"))).as("dist"), sum($"cnt").as("cntSum")).
      withColumn("result", $"cntSum" / (lit(1.0) + $"dist"))

    // 4 上、下三角合并
    val user_ds8 = user_ds3.select("itemidI", "itemidJ", "result").union(user_ds3.select($"itemidJ".as("itemidI"), $"itemidI".as("itemidJ"), $"result"))

    // 5 结果返回
    val out = user_ds8.select("itemidI", "itemidJ", "result").map { row =>
      val itemidI = row.getString(0)
      val itemidJ = row.getString(1)
      val similar = row.getDouble(2)
      ItemSimi(itemidI, itemidJ, similar)
    }
    out
  }

  /**
   * 同现相似度矩阵计算.
   * w(i,j) = N(i)∩N(j)/sqrt(N(i)*N(j))
   * @param user_rdd 用户评分
   * @param RDD[ItemSimi] 返回物品相似度
   *
   */
  def CooccurrenceSimilarity(user_ds: Dataset[ItemPref]): Dataset[ItemSimi] = {
    import user_ds.sparkSession.implicits._

    // 1 (用户：物品) => (用户：(物品集合))
    val user_ds1 = user_ds.groupBy("userid").agg(collect_set("itemid")).withColumnRenamed("collect_set(itemid)", "itemid_set")

    // 2 物品:物品，上三角数据
    val user_ds2 = user_ds1.flatMap { row =>
      val itemlist = row.getAs[scala.collection.mutable.WrappedArray[String]](1).toArray.sorted
      val result = new ArrayBuffer[(String, String, Double)]()
      for (i <- 0 to itemlist.length - 2) {
        for (j <- i + 1 to itemlist.length - 1) {
          result += ((itemlist(i), itemlist(j), 1.0))
        }
      }
      result
    }.withColumnRenamed("_1", "itemidI").withColumnRenamed("_2", "itemidJ").withColumnRenamed("_3", "score")

    // 3 计算物品与物品，上三角,同现频次
    val user_ds3 = user_ds2.groupBy("itemidI", "itemidJ").agg(sum("score").as("sumIJ"))

    // 4 计算物品总共出现的频次
    val user_ds0 = user_ds.withColumn("score", lit(1)).groupBy("itemid").agg(sum("score").as("score"))

    // 5 计算同现相似度
    val user_ds4 = user_ds3.join(user_ds0.withColumnRenamed("itemid", "itemidJ").withColumnRenamed("score", "sumJ").select("itemidJ", "sumJ"), "itemidJ")

    val user_ds5 = user_ds4.join(user_ds0.withColumnRenamed("itemid", "itemidI").withColumnRenamed("score", "sumI").select("itemidI", "sumI"), "itemidI")

    // 根据公式N(i)∩N(j)/sqrt(N(i)*N(j)) 计算
    val user_ds6 = user_ds5.withColumn("result", col("sumIJ") / sqrt(col("sumI") * col("sumJ")))

    // 6 上、下三角合并
    println(s"user_ds6.count(): ${user_ds6.count()}")
    val user_ds8 = user_ds6.select("itemidI", "itemidJ", "result").union(user_ds6.select($"itemidJ".as("itemidI"), $"itemidI".as("itemidJ"), $"result"))
    println(s"user_ds8.count(): ${user_ds8.count()}")
    
    // 7 结果返回
    val out = user_ds8.select("itemidI", "itemidJ", "result").map { row =>
      val itemidI = row.getString(0)
      val itemidJ = row.getString(1)
      val similar = row.getDouble(2)
      ItemSimi(itemidI, itemidJ, similar)
    }
    out
  }

  /**
   * 计算推荐结果.
   * @param items_similar 物品相似矩阵
   * @param user_prefer 用户评分表
   * @param RDD[UserRecomm] 返回用户推荐结果
   *
   */
  def Recommend(items_similar: Dataset[ItemSimi],
                user_prefer: Dataset[ItemPref]): Dataset[UserRecomm] = {
    import user_prefer.sparkSession.implicits._

    //   1 数据准备 
    val items_similar_ds1 = items_similar
    val user_prefer_ds1 = user_prefer
    //   2 根据用户的item召回相似物品
    val user_prefer_ds2 = items_similar_ds1.join(user_prefer_ds1, $"itemidI" === $"itemid", "inner")
    //    user_prefer_ds2.show()
    //   3 计算召回的用户物品得分
    val user_prefer_ds3 = user_prefer_ds2.withColumn("score", col("pref") * col("similar")).select("userid", "itemidJ", "score")
    //    user_prefer_ds3.show()
    //   4 得分汇总
    val user_prefer_ds4 = user_prefer_ds3.groupBy("userid", "itemidJ").agg(sum("score").as("score")).withColumnRenamed("itemidJ", "itemid")
    //    user_prefer_ds4.show()
    //   5 用户得分排序结果，去除用户已评分物品
    val user_prefer_ds5 = user_prefer_ds4.join(user_prefer_ds1, Seq("userid", "itemid"), "left").where("pref is null")
    //    user_prefer_ds5.show()
    //  6 结果返回
    val out1 = user_prefer_ds5.select("userid", "itemid", "score").map { row =>
      val userid = row.getString(0)
      val itemid = row.getString(1)
      val pref = row.getDouble(2)
      UserRecomm(userid, itemid, pref)
    }
    //    out1.orderBy($"userid", $"pref".desc).show
    out1
  }

}


