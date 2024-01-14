#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <vector>
#include <sstream>


// Define a structure to represent an MNIST image
struct MNISTImage
{
    std::vector<double> features; // Vector to store pixel values as features
};

// Structure to store search results
struct SearchResults
{
    std::vector<size_t> indices;
    std::vector<double> trueDistances;
    std::vector<double> approximateDistances;
    std::vector<double> approximationFactors;
};

// Function to format time duration as a string
std::string formatTimeDuration(const std::chrono::microseconds &duration)
{
    auto microseconds = duration.count();
    auto seconds = microseconds / 1000000;
    microseconds %= 1000000;
    auto minutes = seconds / 60;
    seconds %= 60;
    auto hours = minutes / 60;
    minutes %= 60;

    std::ostringstream oss;
    oss << hours << "h " << minutes << "m " << seconds << "s " << microseconds << "microseconds";
    
    
    return oss.str();
}

// Calculate Euclidean distance between two MNIST images //
double calculateDistance(const MNISTImage &img1, const MNISTImage &img2)
{
    double sum = 0.0;
    // Iterate over each feature (pixel) and calculate the squared difference
    for (size_t i = 0; i < img1.features.size(); ++i)
    {
        double diff = img1.features[i] - img2.features[i];
        sum += diff * diff;
    }
    // Return the sqr root for Euclidean distance //
    return std::sqrt(sum);
}

// Function to approximate the distance between two MNIST images based on MRNG_construction //
double approximateDistanceFunction(const MNISTImage &img1, const MNISTImage &img2, const std::set<size_t> &neighbors)
{
    // Maybe better implementation here ---------------------------------------??
    return calculateDistance(img1, img2);
}

// Function to perform search on graph with distances and return results
SearchResults searchOnGraphWithDistances(const std::vector<std::set<size_t>> &graph, size_t start,
    const MNISTImage &query, size_t k, const std::vector<MNISTImage> &mnistDataset,
    size_t candidateLimit)
{
    SearchResults result;
    std::vector<bool> checked(graph.size(), false);
    std::vector<std::pair<size_t, double>> candidates;

    size_t i = 0;
    result.indices.push_back(start);
    checked[start] = true;

    while (i < k)
    {
        size_t current = result.indices.back();

        // Iterate over neighbors of the current node //
        for (size_t neighbor : graph[current])
        {
            if (!checked[neighbor])
            {
                double trueDistance = calculateDistance(query, mnistDataset[neighbor]);
                double approximateDistance =
                    approximateDistanceFunction(mnistDataset[current], mnistDataset[neighbor], graph[current]);

                candidates.emplace_back(neighbor, trueDistance);

                //true and approximate distances//
                result.trueDistances.push_back(trueDistance);
                result.approximateDistances.push_back(approximateDistance);

                // Calculate and store the approximation factor maf
                double approximationFactor = trueDistance/approximateDistance;
                result.approximationFactors.push_back(approximationFactor);
            }
        }

        // Sort the candidates in ascending order of distance to q //
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });

        // Add the closest node to the result //
        result.indices.push_back(candidates[0].first);
        checked[candidates[0].first] = true;
        candidates.clear();

        ++i;
    }

    return result;
}

// fucntion Of MRNG CONSTRUCTION //
std::vector<std::set<size_t>> constructMRNG(const std::vector<MNISTImage> &dataset, size_t candidateLimit)
{
    std::vector<std::set<size_t>> neighbors(dataset.size());

    // Initialize a rng / random number gen//
    std::random_device rd;
    std::mt19937 gen(rd());

    auto startTimeGraphConstruction = std::chrono::high_resolution_clock::now();

    for (size_t p = 0; p < dataset.size(); ++p)
    {
        std::vector<size_t> candidateNeighbors(candidateLimit);
        std::iota(candidateNeighbors.begin(), candidateNeighbors.end(), 0);

        // Shuffle the candidate neighbors randomly //
        std::shuffle(candidateNeighbors.begin(), candidateNeighbors.end(), gen);

        for (size_t q : candidateNeighbors)
        {
            if (p == q)
                continue;

            bool isValidEdge = true;

            for (size_t r : neighbors[p])
            {
                if (calculateDistance(dataset[p], dataset[q]) > calculateDistance(dataset[r], dataset[q]))
                {
                    isValidEdge = false;
                    break;
                }
            }

            if (isValidEdge)
            {
                neighbors[p].insert(q);
            }
        }
    }

    auto endTimeGraphConstruction = std::chrono::high_resolution_clock::now();
    auto elapsedTimeGraphConstruction =
        std::chrono::duration_cast<std::chrono::seconds>(endTimeGraphConstruction - startTimeGraphConstruction).count();

    std::cout << "Graph Construction Time: " << elapsedTimeGraphConstruction << " seconds" << std::endl;

    return neighbors;
}

// Function to find the nearest neighbor on  MRNG //
size_t findNearestNeighborOnGraph(const std::vector<MNISTImage> &dataset, const MNISTImage &query,
                                  const std::vector<std::set<size_t>> &mrngEdges, size_t candidateLimit)
{
    size_t startNode = 0; 
    return searchOnGraphWithDistances(mrngEdges, startNode, query, 1, dataset, candidateLimit).indices[1];
}

// Function to find the top/closest N neighbors on the MRNG //
std::vector<size_t> findTopNNeighborsOnGraph(const std::vector<MNISTImage> &dataset, const MNISTImage &query, const std::vector<std::set<size_t>> &mrngEdges, size_t N,
    size_t candidateLimit)
{
    return searchOnGraphWithDistances(mrngEdges, findNearestNeighborOnGraph(dataset, query, mrngEdges, candidateLimit), query, N, dataset, candidateLimit)
        .indices;
}

// Function to load MNIST_images from a binary file //
std::vector<MNISTImage> loadMNISTImages(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the size of the file/dataset //
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of images based on file size //
    size_t imageSize =4*4*8 + 1; // Each image is 4*4*8 pixels, label is +1 //
    size_t numImages = static_cast<size_t>(fileSize / imageSize);

    // Print the number of images being read //
    std::cout << "Reading " << numImages << " images from the input file." << std::endl;

    // Skip header info //
    file.seekg(16);

    std::vector<MNISTImage> dataset(numImages);
    for (size_t i = 0; i < numImages; ++i)
    {
        dataset[i].features.resize(4*4*8);

        for (int j = 0; j <4*4*8; ++j)
        {
            uint8_t pixelValue;
            file.read(reinterpret_cast<char *>(&pixelValue), sizeof(pixelValue));
            dataset[i].features[j] = static_cast<double>(pixelValue) / 255.0;
        }
    }

    return dataset;
}

//visualize the MRNG //
void visualizeMRNG(const std::vector<std::set<size_t>> &neighbors, const std::string &filename)
{
    std::ofstream dotFile(filename);

    if (!dotFile.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    dotFile << "graph MRNG {" << std::endl;

    for (size_t p = 0; p < neighbors.size(); ++p)
    {
        for (size_t r : neighbors[p])
        {
            if (p < r)
            {
                dotFile << "  " << p << " -- " << r << ";" << std::endl;
            }
        }
    }

    dotFile << "}" << std::endl;
    dotFile.close();
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " input_file query_file numNeighbors candidateLimit output_file"
                  << std::endl;
        return EXIT_FAILURE;
    }


    std::string inputFilename = argv[1];
    std::string queryFilename = argv[2];
    size_t numNeighbors = std::stoi(argv[3]);
    size_t candidateLimit = std::stoi(argv[4]);
    std::string outputFilename = argv[5];

    std::vector<MNISTImage> mnistDataset = loadMNISTImages(inputFilename);

    auto startTimeTotal = std::chrono::high_resolution_clock::now();

    std::vector<std::set<size_t>> mrngEdges = constructMRNG(mnistDataset, candidateLimit);

    std::ifstream queryFile(queryFilename, std::ios::binary);
    if (!queryFile.is_open())
    {
        std::cerr << "Error opening query file." << std::endl;
        return EXIT_FAILURE;
    }

    auto startTimeSearch = std::chrono::high_resolution_clock::now();

    std::ofstream outputFile(outputFilename);
    std::streambuf *originalStdout = std::cout.rdbuf();
    std::cout.rdbuf(outputFile.rdbuf());

    double totalApproximationFactor = 0.0;

      for (size_t queryIndex = 0; queryIndex < mnistDataset.size(); ++queryIndex)
    {
        MNISTImage queryImage;
        queryImage.features.resize(4*4*8);

        for (int i = 0; i < 4*4*8; ++i)
        {
            uint8_t pixelValue;
            queryFile.read(reinterpret_cast<char *>(&pixelValue), sizeof(pixelValue));
            queryImage.features[i] = static_cast<double>(pixelValue) / 255.0;
        }

        auto startQueryTime = std::chrono::high_resolution_clock::now();

        size_t nearestNeighborOnGraph = findNearestNeighborOnGraph(mnistDataset, queryImage, mrngEdges, candidateLimit);
        double nearestNeighborDistanceOnGraph = calculateDistance(queryImage, mnistDataset[nearestNeighborOnGraph]);

        std::cout << "Query " << queryIndex + 1
                  << ": Nearest Neighbor Index (Search on Graph): " << nearestNeighborOnGraph
                  << ", Distance: " << nearestNeighborDistanceOnGraph << std::endl;

        SearchResults searchResults = searchOnGraphWithDistances(mrngEdges, nearestNeighborOnGraph, queryImage, numNeighbors, mnistDataset, candidateLimit);

        auto endQueryTime = std::chrono::high_resolution_clock::now();
        auto elapsedQueryTime = std::chrono::duration_cast<std::chrono::microseconds>(endQueryTime - startQueryTime).count();

        std::cout << "Query Time: " << elapsedQueryTime << " microseconds" << std::endl;

        double totalTrueDistanceTime = std::accumulate(searchResults.trueDistances.begin(), searchResults.trueDistances.end(), 0.0);
        double totalApproximateDistanceTime = std::accumulate(searchResults.approximateDistances.begin(), searchResults.approximateDistances.end(), 0.0);

        // Calculate average times for the current query for the average //
        double averageTrueDistanceTime = totalTrueDistanceTime / numNeighbors;
        double averageApproximateDistanceTime = totalApproximateDistanceTime / numNeighbors;

        // Format times for display //
        std::string formattedAverageTrueDistanceTime = formatTimeDuration(std::chrono::microseconds(static_cast<long long>(averageTrueDistanceTime)));
        std::string formattedAverageApproximateDistanceTime = formatTimeDuration(std::chrono::microseconds(static_cast<long long>(averageApproximateDistanceTime)));

        std::cout << "Average True Distance Time: " << formattedAverageTrueDistanceTime << std::endl;
        std::cout << "Average Approximate Distance Time: " << formattedAverageApproximateDistanceTime << std::endl;

        std::cout << "Top " << numNeighbors << " Neighbors (Search on Graph):" << std::endl;
        for (size_t i = 0; i < numNeighbors && i < searchResults.indices.size(); ++i)
        {
            size_t neighborIndex = searchResults.indices[i];
            double trueDistance = searchResults.trueDistances[i];
            double approximateDistance = searchResults.approximateDistances[i];
            double approximationFactor = searchResults.approximationFactors[i];

            std::cout << "Index: " << neighborIndex << ", True Distance: " << trueDistance
                      << ", Approximate Distance: " << approximateDistance << ", Approximation Factor: " << approximationFactor << std::endl;

            // Accumulate the maximum approximation factor for each query
            totalApproximationFactor += approximationFactor;
        }

        std::cout << "Maximum Approximation Factor for Query " << queryIndex + 1 << ": " << *std::max_element(searchResults.approximationFactors.begin(), searchResults.approximationFactors.end()) << std::endl;
    }

    // Calculate and print the average approximation factor across all queries
    double averageApproximationFactor = totalApproximationFactor / (mnistDataset.size()*numNeighbors);
    std::cout << "Average Approximation Factor across all queries: " << averageApproximationFactor << std::endl;

    std::cout.rdbuf(originalStdout);

    auto endTimeSearch = std::chrono::high_resolution_clock::now();
    auto elapsedTimeSearch = std::chrono::duration_cast<std::chrono::seconds>(endTimeSearch - startTimeSearch).count();

    std::cout << "Search Execution Time is : " << elapsedTimeSearch << "seconds" << std::endl;
    visualizeMRNG(mrngEdges, "mrng_visualization.dot");

    auto endTimeTotal = std::chrono::high_resolution_clock::now();
    auto elapsedTimeTotal = std::chrono::duration_cast<std::chrono::seconds>(endTimeTotal - startTimeTotal).count();

    std::cout << "Total Execution Time is:" << elapsedTimeTotal << " seconds" << std::endl;

    return 0;
}