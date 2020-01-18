  /**
   * 读取libSVM格式的文件，生成训练样本和测试样本。
   * 1）读取文件
   * 2）生成标签索引
   * 3）样本处理
   * 4）样本划分
   */
  def readLibSvmSampleData(
    @transient spark: org.apache.spark.sql.SparkSession,
    dataPath: String): (Dataset[LabeledPoint], Dataset[LabeledPoint]) = {
    import spark.implicits._
    // 2.1 读取样本
    val dataRead = spark.read.options(Map(("delimiter", "|"), ("header", "false"))).csv(dataPath)
    // 2.2 获取样本中所有标签，并且建立索引关系    
    val featureMap = dataRead.map {
      case Row(libSvmFeatrue: String) =>
        val items = libSvmFeatrue.split(' ')
        val features = items.filter(_.nonEmpty).
          filter(f => f.split(':').size == 2).
          map { item =>
            val indexAndValue = item.split(':')
            indexAndValue(0)
          }
        features
    }.flatMap(x => x).distinct().collect().sorted.zipWithIndex.toMap
    val numFeatures = featureMap.size
    // 2.3 样本校准化处理
    val readSampleData = dataRead.map {
      case Row(libSvmFeatrue: String) =>
        val items = libSvmFeatrue.split(' ')
        val click = items(0).toString().toDouble
        val features = items.filter(_.nonEmpty).
          filter(f => f.split(':').size == 2).
          map { item =>
            val indexAndValue = item.split(':')
            val id = featureMap.getOrElse(indexAndValue(0), -1)
            val value = indexAndValue(1).toDouble
            (id, value)
          }.filter(f => f._1 > 0).sortBy(f => f._1)
        val label = if (click > 0) 1.0 else 0.0
        LabeledPoint(label, Vectors.sparse(numFeatures, features.map(_._1), features.map(_._2)))
    }
    // 2.3 划分样本
    val splits = readSampleData.randomSplit(Array(0.6, 0.4))
    val training = splits(0)
    val test = splits(1)
    (training, test)
  }
  
  /**
   * 根据gbdt模型对样本进行转换生成新样本
   * 每个样本通过每一棵树，可以找到对应的叶节点，该叶节点就是转换后的新特征。
   * @param sampleLablePoint 训练样本，格式为：RDD[LabeledPoint].
   * @param treeLeafMap gbdt模型的叶子节点.
   * @param gbdtModel gbdt模型
   * @return RDD[LabeledPoint]
   */
  def lrSample(
    sampleLablePoint: RDD[LabeledPoint],
    lrFeatureMap: Map[String, Int],
    gbdtModel: GradientBoostedTreesModel): RDD[LabeledPoint] = {
    val treeNumber = gbdtModel.trees.length
    val lrFeatureNum = lrFeatureMap.size
    val lrSampleParsed = sampleLablePoint.map { point =>
      val label = point.label
      val features = point.features
      val lrFeatures = ArrayBuffer[Int]()
      val lrValues = ArrayBuffer[Double]()
      val treeNumber = gbdtModel.trees.size
      for (treeIndex <- 0 to (treeNumber - 1)) {
        var node = gbdtModel.trees(treeIndex).topNode
        while (!node.isLeaf) {
          if (node.split.get.featureType == Continuous) {
            if (features(node.split.get.feature) <= node.split.get.threshold)
              node = node.leftNode.get
            else
              node = node.rightNode.get
          } else {
            if (node.split.get.categories.contains(features(node.split.get.feature)))
              node = node.leftNode.get
            else
              node = node.rightNode.get
          }
        }
        val key = treeIndex.toString + '_' + node.id

        lrFeatures += lrFeatureMap(key)
        lrValues += 1
      }
      (label, lrFeatures.sorted.toArray, lrValues.toArray)
    }
    val lrSamplLablePoint = lrSampleParsed.map {
      case (label, lrFeatures, lrValues) =>
        LabeledPoint(label, Vectors.sparse(lrFeatureNum, lrFeatures, lrValues))
    }
    (lrSamplLablePoint)
  }

  /**
   * gbdt模型解析叶子节点
   * @param gbdtModel gbdt模型.
   * @return 返回Map[String, Int]，得到所有决策树的叶子节点，以及编号，数据格式为：(树id_叶子节点id, 编号)
   */
  def getTreeLeafMap(gbdtModel: GradientBoostedTreesModel): Map[String, Int] = {
    val lrFeatureMap = scala.collection.mutable.Map[String, Int]()
    var featureId = 0
    val treeNumber = gbdtModel.trees.size
    for (treeIndex <- 0 to (treeNumber - 1)) {
      val treeNodeQueue = collection.mutable.Queue[Node]()
      val rootNode = gbdtModel.trees(treeIndex).topNode
      treeNodeQueue.enqueue(rootNode)
      while (!treeNodeQueue.isEmpty) {
        val resNode = treeNodeQueue.dequeue()
        if (resNode.isLeaf) {
          val key = treeIndex.toString + '_' + resNode.id.toString()
          lrFeatureMap(key) = featureId
          featureId = featureId + 1
        }
        if (resNode.leftNode.isDefined)
          treeNodeQueue.enqueue(resNode.leftNode.get)
        if (resNode.rightNode.isDefined)
          treeNodeQueue.enqueue(resNode.rightNode.get)
      }
    }
    (lrFeatureMap.toMap)
  }

  