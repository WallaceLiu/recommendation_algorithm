package book_code

import scala.collection.mutable.ArrayBuffer

class myGoodsRecall extends Serializable {

  /**
   *
   * 根据用户请求，返回召回列表
   *
   */

  def recall(request: Request, extendMap: Map[String, String]): Response = {

    // 1 获取参数
    val recallNum = extendMap.getOrElse("recallNum", "500").toInt
    val recallByKeyNum = extendMap.getOrElse("recallByKeyNum", "20").toInt
    val userActTopK = extendMap.getOrElse("userActTopK", "20").toInt

    // 2.1 获取用户数据，取用户TopK个浏览商品，这里一般通过其他接口，取相应的用户数据，在代码中不展开，这里采用一个数组来做实例讲解
    val userGoldsdArray = Array("101", "108", "109", "105")
    // 2.2 获取用户数据，取用户类别兴趣数据，这里一般通过其他接口，取相应的用户数据，在代码中不展开，这里采用一个数组来做实例讲解
    val userCategoryArray = Array("1", "2", "11")

    // 3.1 goldCF召回查询
    val userGoldCfRecallArray = userGoldsdArray.map { itemKey: String =>
      // 通过key查询，得到列表，这里一般通过其他接口，取相应的数据，在本代码中不展开，这里采用一个Map
      // 需要解析召回内容，并且取top，用数据格式返回
      val itemByOneKeyArray = Map[String, Array[Item]]().getOrElse(itemKey, Array[Item]()).slice(0, recallByKeyNum)
      itemByOneKeyArray
    }
    // 3.2 汇总并去重
    val userGoldCfRecallDistinctTmp = userGoldCfRecallArray.flatMap(f => f)
    val userGoldCfRecallDistinct = ArrayBuffer[Item]()
    for (i <- 0 to userGoldCfRecallDistinctTmp.size - 1) {
      val item = userGoldCfRecallDistinctTmp(i)
      if (!userGoldCfRecallDistinct.map(f => f.itemKey).contains(item.itemKey)) {
        userGoldCfRecallDistinct += item
      }
    }

    // 4 相似内容召回查询
    val userGoldSimilarContentArray = userGoldsdArray.map { itemKey: String =>
      // 通过key查询，得到列表，这里一般通过其他接口，取相应的数据，在本代码中不展开，这里采用一个Map来做实例讲解
      // 需要解析召回内容，并且取top，用数据格式返回
      val itemByOneKeyArray = Map[String, Array[Item]]().getOrElse(itemKey, Array[Item]()).slice(0, recallByKeyNum)
      itemByOneKeyArray
    }
    // 4.2 汇总并去重
    val userGoldSimilarContentRecallDistinctTmp = userGoldSimilarContentArray.flatMap(f => f)
    val userGoldSimilarContentRecallDistinct = ArrayBuffer[Item]()
    for (i <- 0 to userGoldSimilarContentRecallDistinctTmp.size - 1) {
      val item = userGoldSimilarContentRecallDistinctTmp(i)
      if (!userGoldSimilarContentRecallDistinctTmp.map(f => f.itemKey).contains(item.itemKey)) {
        userGoldSimilarContentRecallDistinct += item
      }
    }

    // 5 用户类别兴趣召回查询
    val userGoldSimilarCategoryArray = userCategoryArray.map { category: String =>
      // 通过key查询，得到列表，这里一般通过其他接口，取相应的数据，在本代码中不展开，这里采用一个Map来做实例讲解
      // 需要解析召回内容，并且取top，用数据格式返回
      val itemByOneKeyArray = Map[String, Array[Item]]().getOrElse(category, Array[Item]()).slice(0, recallByKeyNum)
      itemByOneKeyArray
    }
    // 5.2 汇总并去重
    val userGoldSimilarCategoryRecallDistinctTmp = userGoldSimilarCategoryArray.flatMap(f => f)
    val userGoldSimilarCategoryRecallDistinct = ArrayBuffer[Item]()
    for (i <- 0 to userGoldSimilarCategoryRecallDistinctTmp.size - 1) {
      val item = userGoldSimilarCategoryRecallDistinctTmp(i)
      if (!userGoldSimilarCategoryRecallDistinctTmp.map(f => f.itemKey).contains(item.itemKey)) {
        userGoldSimilarCategoryRecallDistinct += item
      }
    }

    // 6 依此类推，查询其它召回数据，这里主不展开了

    // 7 多个召回数据合并，排序，并且取TopK
    // 7.1 CF
    // 取每个召回的参数权重，这里用个Map来做实例讲解
    val weightCF = Map[String, Double]().getOrElse("CF", 1.0)
    // 取物品，以及对应的分值
    val recallCF = userGoldCfRecallDistinct.toArray.map(x => (x.itemKey, x.score * weightCF))
    // 7.2 Content
    // 取每个召回的参数权重，这里用个Map来做实例讲解
    val weightContent = Map[String, Double]().getOrElse("Content", 1.0)
    // 取物品，以及对应的分值
    val recallContent = userGoldSimilarContentRecallDistinct.toArray.map(x => (x.itemKey, x.score * weightContent))
    // 7.3 Category
    // 取每个召回的参数权重，这里用个Map来做实例讲解
    val weightCategory = Map[String, Double]().getOrElse("Category", 1.0)
    // 取物品，以及对应的分值
    val recallCategory = userGoldSimilarCategoryRecallDistinct.toArray.map(x => (x.itemKey, x.score * weightCategory))

    // 7.4 合并，并且返回ToK，排序按照分值降序排
    val recallMerge = (recallCF ++ recallContent ++ recallCategory).
      sortBy(f => -1 * f._2).
      slice(0, recallNum).map {
        case (itemKey: String, score: Double) =>
          new Item(itemKey).setScore(score)
      }

    // 8 返回结果
    val recallStatus = if (recallMerge.size > 0) "True" else "False"
    val response = new Response(request.getSessionID).
      setStatus(recallStatus).
      setItemArray(recallMerge)
    response
  }

}