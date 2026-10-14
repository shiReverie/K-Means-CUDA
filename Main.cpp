#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cctype>
#include <clocale>
#include <locale>
#include <set>
#include <iomanip>


static std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

static bool startsWithIgnoreCase(const std::string& s, const std::string& prefix) {
    if (s.size() < prefix.size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) return false;
    }
    return true;
}

static bool parseDouble(const std::string& s, double& out) {
    if (s.empty() || s == "?") return false;
    std::istringstream iss(s);
    iss.imbue(std::locale::classic());
    iss >> out;
    return !iss.fail() && iss.eof();
}

static std::vector<std::vector<double>> LoadArffData(const std::string& filePath) {
    std::vector<std::vector<double>> data;
    std::ifstream fin(filePath);
    if (!fin.is_open()) {
        std::cerr << "Could not open: " << filePath << "\n";
        return data;
    }

    bool dataSection = false;
    int expectedAttributes = -1;
    int lineNo = 0;
    int rejected = 0;

    std::string line;
    while (std::getline(fin, line)) {
        ++lineNo;

        if (lineNo == 1 && line.size() >= 3 &&
            (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB &&
            (unsigned char)line[2] == 0xBF) {
            line.erase(0, 3);
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '%') continue;

        if (!dataSection) {
            if (startsWithIgnoreCase(trimmed, "@data"))
                dataSection = true;
            continue;
        }

        std::vector<std::string> parts;
        {
            std::string tok;
            for (char ch : trimmed) {
                if (ch == ',' || std::isspace(static_cast<unsigned char>(ch))) {
                    if (!tok.empty()) { parts.push_back(tok); tok.clear(); }
                }
                else {
                    tok.push_back(ch);
                }
            }
            if (!tok.empty()) parts.push_back(tok);
        }
        if (parts.empty()) continue;

        std::vector<double> values;
        values.reserve(parts.size());
        bool ok = true;

        for (const auto& p : parts) {
            if (p.empty()) continue;                 
            if (p == "?") { ok = false; break; }     
            double v;
            if (!parseDouble(p, v)) { ok = false; break; }
            values.push_back(v);
        }
        if (!ok || values.empty()) { ++rejected; continue; }

        if (expectedAttributes == -1)
            expectedAttributes = static_cast<int>(values.size());

        if (static_cast<int>(values.size()) == expectedAttributes) {
            data.push_back(std::move(values));
        }
        else {
            ++rejected;
        }
    }

    std::cout << "Rows read: " << data.size()
        << ", rejected: " << rejected << "\n";
    if (expectedAttributes > 0)
        std::cout << "Attributes per row: " << expectedAttributes << "\n";

    return data;
}


static std::vector<int> KMeans(const std::vector<std::vector<double>>& data,
    int k, int maxIterations) {
    const int n = static_cast<int>(data.size());
    const int dims = static_cast<int>(data[0].size());

    std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
    std::uniform_int_distribution<int> dist(0, n - 1);

    std::vector<std::vector<double>> centroids(k);
    std::vector<int> labels(n, 0);

    for (int i = 0; i < k; ++i)
        centroids[i] = data[dist(rng)];

    for (int iter = 0; iter < maxIterations; ++iter) {
        bool changed = false;

        for (int i = 0; i < n; ++i) {
            int bestCluster = 0;
            double bestDist = std::numeric_limits<double>::max();

            for (int c = 0; c < k; ++c) {
                double d2 = 0.0;
                for (int d = 0; d < dims; ++d) {
                    double diff = data[i][d] - centroids[c][d];
                    d2 += diff * diff;
                }
                if (d2 < bestDist) {
                    bestDist = d2;
                    bestCluster = c;
                }
            }

            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
        }

        if (!changed) break;

        std::vector<std::vector<double>> newCentroids(k,
            std::vector<double>(dims, 0.0));
        std::vector<int> counts(k, 0);

        for (int i = 0; i < n; ++i) {
            int cluster = labels[i];
            ++counts[cluster];
            for (int d = 0; d < dims; ++d)
                newCentroids[cluster][d] += data[i][d];
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            for (int d = 0; d < dims; ++d)
                newCentroids[c][d] /= counts[c];
        }

        centroids.swap(newCentroids);
    }

    return labels;
}


static void PrintResult(const std::vector<int>& labels, int k,
    const std::vector<std::vector<double>>& data) {
    std::cout << "Cluster distribution:\n";
    std::cout << std::string(50, '-') << "\n";

    const int n = static_cast<int>(labels.size());
    int totalAssigned = 0;

    for (int c = 0; c < k; ++c) {
        int count = 0;
        for (int v : labels) if (v == c) ++count;

        if (count > 0) {
            std::cout << "Cluster "
                << std::setw(2) << c << ": "
                << std::setw(5) << count << " items ("
                << std::fixed << std::setprecision(1)
                << (count * 100.0 / n) << "%)\n";
            totalAssigned += count;
        }
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << "Total assigned: " << totalAssigned
        << " of " << n << "\n";

    std::set<int> used(labels.begin(), labels.end());
    int empty = k - static_cast<int>(used.size());
    if (empty > 0)
        std::cout << "Empty clusters: " << empty << "\n";

    std::cout << "\nFirst 10 points:\n";
    std::cout << std::string(70, '-') << "\n";

    int limit = std::min(10, static_cast<int>(data.size()));
    for (int i = 0; i < limit; ++i) {
        std::ostringstream vectorStr;
        int takeDims = std::min(5, static_cast<int>(data[i].size()));
        for (int d = 0; d < takeDims; ++d) {
            if (d > 0) vectorStr << ", ";
            vectorStr << std::fixed << std::setprecision(6) << data[i][d];
        }
        std::cout << "#" << std::setw(3) << i
            << " [" << vectorStr.str() << ", ...]"
            << " -> cluster " << labels[i] << "\n";
    }
}



int main(int argc, char** argv) {
    setlocale(LC_ALL, "C");   

    std::string filePath = (argc > 1) ? argv[1] : "10k20.arff";
    int k = 21;
    int maxIterations = 100;

    std::cout << "K-Means clustering\n";
    std::cout << "File path: " << filePath << "\n";

    std::ifstream test(filePath);
    bool exists = test.good();
    test.close();

    std::cout << "File exists: " << (exists ? "true" : "false") << "\n\n";

    if (!exists) {
        std::cout << "Error: file not found!\n";
        std::cout << "Place 10k20.arff in the program folder\n";
        std::cin.get();
        return 0;
    }

    std::cout << "Loading data from " << filePath << "...\n";
    std::vector<std::vector<double>> data = LoadArffData(filePath);

    if (data.empty()) {
        std::cout << "Error: could not load data!\n";
        std::cin.get();
        return 0;
    }

    int n = static_cast<int>(data.size());
    int dims = static_cast<int>(data[0].size());

    std::cout << "Loaded: " << n << " objects x " << dims << " features\n";
    std::cout << "Parameters: k=" << k << ", max iterations=" << maxIterations << "\n";
    std::cout << "Running K-Means...\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> labels = KMeans(data, k, maxIterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Execution time: " << ms << " ms\n\n";

    PrintResult(labels, k, data);

    std::cout << "\nPress Enter to exit...\n";
    std::cin.get();

    return 0;
}