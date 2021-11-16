import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source
import org.apache.spark.rdd.RDD
object WordCount {
  def main(args: Array[String]): Unit = {
    //不使用spark
    val s = Source.fromFile("D://test.txt")
    val lines=s.getLines().toList.map(_.toLowerCase)
    s.close()
    val res0=lines.map(_.split(" ")).flatten
    //可使用遍历相加代替压扁
    val res1=res0.map(x=>(x,1))
    val res2=res1.groupBy(_._1)
    val res3=res2.mapValues(_.map(_._2).sum).toList.sortBy(_._2)
    res3.foreach(x=>println(x._1+" "+x._2))
    //使用spark
    val conf=new SparkConf().setAppName("wjwWordCount").setMaster("local[4]")
    val sc=new SparkContext(conf)
    //将文本内容输入rdd
    val words=sc.textFile("D://test.txt")
    val result=words.flatMap(_.split(" ")).map(_.toLowerCase).map((_,1)).groupByKey().map(x=>(x._1,x._2.sum)).sortBy(_._2)
    result.foreach(x=>println(x._1+" "+x._2))
  }
}
