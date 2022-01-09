import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import scala.collection.mutable.ListBuffer

// Reading in the data from the local file path
val rawData = sc.textFile ("/home/rel/library/data/researchdata_cup/researchdatacup.datay")

// Alternatively via HDFS.
// val rawData = sc.textFile ("hdfs:/user/Azeroual/data/researchdatacup.data") 

val symbolData = Map(
" protocols " -> rawData.map(_.split(’,’)).map(_(1)).distinct.collect.zipWithIndex.toMap,
" services " -> rawData.map(_.split (’,’)).map(_(2)).distinct.collect.zipWithIndex.toMap,
" states " -> rawData.map(_.split(’,’)).map(_(3)).distinct.collect.zipWithIndex.toMap
)

// Returns an RDD from feature vectors
def getFeatureVectors (rawData: RDD [String], symbolData: Map [String, 
Map [String, Int]]): RDD [Vector] = {
rawData .map {line =>
// For each "line", create a buffer
val buffer = line.split (’,’).toBuffer

// Extract protocol, services, status and identification information from the list
val protocol = buffer.remove (1)
val service = buffer.remove (1)
val state = buffer.remove (1)

// "Type of connection" ("normal" etc., last entry)
// remove as this information should not be used here
buffer.remove(buffer.length - 1) 

// Create a vector with the rest of the values from the list
val vector = buffer.map (_.toDouble)

// Create numeric feature arrays for protocols, services and states
val protocolFeatures = new Array [Double] (symbolData ("protocols").size)
val serviceFeatures = new Array [Double] (symbolData ("services").size)
val stateFeatures = new Array [Double] (symbolData ("states") .size)

// Set the value to 1.0 for the respective feature (One-Hot)
protocolFeatures (symbolData ("protocols") (protocol)) = 1.0
serviceFeatures (symbolData ("services") (service)) = 1.0
stateFeatures (symbolData ("states") (state)) = 1.0

// Add the features to the vector
vector. insertAll (1, stateFeatures)
vector. insertAll (1, serviceFeatures)
vector. insertAll (1, protocolFeatures)

// Create a vector that can be used for k-means from the assembled "vector"
Vectors.dense(vector.ToArray)
}
} 

val vectorRdd = getFeatureVectors (rawData, symbolData).cache()
val kMeansModel = (new KMeans()).run(vectorRdd)

// Calculates the distance between two vectors using the Euclidean distance: square root (sum (i=1 to n){(x[i] - y[i])^2})

def euclideanDistance (x:Vector, y:Vector) = {
math.sqrt (
x. toArray // Convert vector x and y to an array
.zip (y. ToArray) // Array (0, 1) .zip (Array (’a’, ’b’)) -> Array ((0, ’a’), (1, ’b’))
.map (i => i._1 - i._2) // Corresponds to: x[i] - y[i], _1 is the first value of the vector tuple "i", _2 the second
.map (j => j * j) // Corresponds to: (x[i] - y[i]) ^ 2
.sum // Sum previous records
)
} 

// Determines the cluster for a feature vector and its next cluster centroid and calculates the distance between the two vectors
def distanceToCentroid (featureVector:Vector, kMeansModel:KMeansModel) = {
val calculatedCluster = kMeansModel.predict(featureVector)
val nextCentroid = kMeansModel.clusterCenters(calculatedCluster)
euclideanDistance(nextCentroid,featureVector)
}

// Create array from feature vectors
val featuresArray = vectorRdd.map(_.toArray)

// Determine the number of feature columns
val numberFeaturesPerDataset = featuresArray.first.length

// Determine the number of data records
val numberDatasets = featuresArray.count

// Add up all feature values ​​to one another. Result: array with sums of all feature columns
val featureSums = featuresArray.reduce {(x, y) =>
val featureTuple = x.zip(y) // Create tuple from (feature, feature +1)
featureTuple.map(tuple => tuple._1 + tuple._2) // Add both tuple objects
}

// Calculate the sums of squares
val sumOfSquares = featuresArray.fold(new Array[Double](numberFeaturesPerDataset)) {(x, y) =>
val featureTuple = x.zip(y)
featureTuple.map(tuple => tuple._1 + tuple._2 * tuple._2)
}

// Calculation of the standard deviations 
val standardDeviations = sumOfSquares.zip(featureSums).map {case (sumSquare,featureSum) =>
math.sqrt(numberDatasets * sumSquare - featureSum * featureSum) / numberDatasets
}

// Calculation of the mean values
val featureMeans = featureSums.map (_ / numberDatasets)

// Calculation of the z-transformation for a vector
def zTransform (featureVector:Vector, featureMeans:Array[Double],standardDeviations:Array[Double]) = {
val transformedArray = (featureVector.toArray,featureMeans, 
standardDeviations).zipped.map{case(featureVector,featureMean,standardDeviation) =>
// Avoid division by 0:
if (standardDeviation <= 0) {
(featureVector - featureMean)
} else {
(featureVector - featureMean) / standardDeviation
}
}

// Generate feature vectors from "transformedArray"
Vectors.dense(transformedArray)
}

// Standardize feature vectors
val standardizedVectorRdd = vectorRdd.map(zTransform (_,featureMeans,standardDeviations)).cache()

// Create a changeable list "anomalyRates"
val anomalyRates = new ListBuffer[(Int, Double)]()

// As long as k is between 10 and 200, increment by 10 each time
for (k <- 10 to 210 by 10) {
// Create new k-means model for "standardizedVectorRdd" with specific number of clusters from "k"
val kMeansModel = (new KMeans())
.setK(k)
.setEpsilon(1.0e -6)
.run(standardizedVectorRdd)

// Calculate distance threshold with new k-means model.
val distanceThreshold = standardizedVectorRdd.map{distanceToCentroid(_, kMeansModel)}.mean()

val dataFeatureVectorTuple = rawData.zip(standardizedVectorRdd)
val anomalies = dataFeatureVectorTuple.filter{case(data,featureVector) => 
distanceToCentroid(featureVector,kMeansModel) > distanceThreshold}.keys

val anomalyRate = (anomalies.count.toDouble / standardizedVectorRdd.count.toDouble) * 100

// Add to the anomaly rates list
val kRateTuple = (k, anomalyRate)
anomalyRates + = kRateTuple} 

// k = 100 seems like a good value, create kMeansModel for that
val kMeansModel = (new KMeans())
.setK(100)
.setRuns(5)
.setEpsilon(1.0e -6)
.run(standardizedVectorRdd)

val distanceThreshold = standardizedVectorRdd.map{distanceToCentroid(_, kMeansModel)}.mean() 

val dataFeatureVectorTuple = rawData.zip(standardizedVectorRdd)
val anomalies = dataFeatureVectorTuple .filter{case (data,featureVector) =>
distanceToCentroid (featureVector, kMeansModel) > distanceThreshold}.keys

// Sorted output of the number of all anomalies detected per type of attack
anomalies.map (_.split (",").last).countByValue.toSeq.sortBy (_._ 2).reverse.foreach(println)

// Output one hundred thousandth of all anomalies.
anomalies.sample(false, 0.00001).foreach(println)

// Anomaly rates for Spark SQL
case class Rates (k:Integer, rate:Double)
anomalyRates .map {case (k, rate) => Rates (k, rate)}.toDF.registerTempTable ("anomalyRates")
 
 
 
 
// Spark SQL query:
// SELECT k, rate FROM anomalyRates 



