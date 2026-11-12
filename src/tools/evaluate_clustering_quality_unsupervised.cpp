// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#include <getopt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using point_id_t   = std::size_t;
using cluster_id_t = long long;
using graph_t      = std::unordered_map<point_id_t, std::vector<point_id_t>>;
using points_t     = std::vector<std::vector<double>>;

struct options_t {
  std::filesystem::path points_path;
  std::filesystem::path graph_path;
  std::filesystem::path cluster_path;
  std::string           knng_format;
  std::string           distance_function;
  std::string           algorithm = "bfs";
  std::filesystem::path output_path;
  double                epsilon         = 0.001;
  bool                  points_have_ids = false;
  bool                  verbose         = false;
};

struct match_t {
  point_id_t  point_id;
  std::size_t hops;
  double      distance;
};

struct result_t {
  point_id_t             point_id;
  cluster_id_t           cluster_id;
  std::optional<match_t> closest_in;
  std::optional<match_t> closest_out;
};

using labels_t = std::unordered_map<point_id_t, cluster_id_t>;

[[noreturn]] void usage_error(const std::string &message) {
  throw std::invalid_argument(
      message +
      "\nUsage: evaluate_clustering_quality_unsupervised "
      "-g GRAPH --knng-format {M,MI,B,BI} -c CLUSTERS "
      "[-p POINTS] [-I] [-f {l2,l1,cosine}] [-e EPSILON] "
      "[-a {bfs,dijkstra}] [-o OUTPUT] [-v]");
}

void print_usage(std::ostream &out) {
  out << "Usage: evaluate_clustering_quality_unsupervised "
         "-g GRAPH --knng-format {M,MI,B,BI} -c CLUSTERS "
         "[-p POINTS] [-I] [-f {l2,l1,cosine}] "
         "[-e EPSILON] [-a {bfs,dijkstra}] [-o OUTPUT] [-v]\n"
      << "  -p, --points PATH       Feature vectors file or directory "
         "(dijkstra only)\n"
      << "  -I, --ids               Feature-vector rows include point IDs\n"
      << "  -g, --graph PATH        kNNG file or directory\n"
      << "      --knng-format FMT   Distance-bearing format: M, MI, B, or BI\n"
      << "  -f, --dist_func NAME   l2, l1, or cosine (dijkstra only)\n"
      << "  -c, --cluster PATH     Point ID/cluster ID file\n"
      << "  -e, --epsilon VALUE    Positive log-score offset (default: 0.001)\n"
      << "  -a, --algorithm NAME  bfs (scores use hops) or dijkstra (default: "
         "bfs)\n"
      << "  -o, --output PATH      Output prefix (otherwise print cluster "
         "scores)\n"
      << "  -v, --verbose          Enable verbose output\n";
}

options_t parse_options(int argc, char **argv) {
  options_t    options;
  const option long_options[] = {
      {"points", required_argument, nullptr, 'p'},
      {"ids", no_argument, nullptr, 'I'},
      {"graph", required_argument, nullptr, 'g'},
      {"knng-format", required_argument, nullptr, 1000},
      {"dist_func", required_argument, nullptr, 'f'},
      {"cluster", required_argument, nullptr, 'c'},
      {"epsilon", required_argument, nullptr, 'e'},
      {"algorithm", required_argument, nullptr, 'a'},
      {"output", required_argument, nullptr, 'o'},
      {"verbose", no_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  opterr           = 0;
  int option_index = 0;
  while (true) {
    const int value = getopt_long(argc, argv, "p:Ig:f:c:e:a:o:vh", long_options,
                                  &option_index);
    if (value == -1) break;
    switch (value) {
      case 'p':
        options.points_path = optarg;
        break;
      case 'I':
        options.points_have_ids = true;
        break;
      case 'g':
        options.graph_path = optarg;
        break;
      case 1000:
        options.knng_format = optarg;
        break;
      case 'f':
        options.distance_function = optarg;
        break;
      case 'c':
        options.cluster_path = optarg;
        break;
      case 'e': {
        std::size_t consumed = 0;
        options.epsilon      = std::stod(optarg, &consumed);
        if (consumed != std::string(optarg).size()) {
          usage_error("Invalid epsilon value");
        }
        break;
      }
      case 'a':
        options.algorithm = optarg;
        break;
      case 'o':
        options.output_path = optarg;
        break;
      case 'v':
        options.verbose = true;
        break;
      case 'h':
        print_usage(std::cout);
        std::exit(EXIT_SUCCESS);
      default:
        usage_error("Unknown or incomplete option");
    }
  }

  if (optind != argc) usage_error("Unexpected positional argument");
  if (options.graph_path.empty()) usage_error("Graph path is required");
  if (options.cluster_path.empty()) usage_error("Cluster path is required");
  if (options.knng_format != "M" && options.knng_format != "MI" &&
      options.knng_format != "B" && options.knng_format != "BI") {
    usage_error("--knng-format must be one of M, MI, B, or BI");
  }
  if (options.algorithm != "bfs" && options.algorithm != "dijkstra") {
    usage_error("Algorithm must be bfs or dijkstra");
  }
  if (!options.distance_function.empty() && options.distance_function != "l1" &&
      options.distance_function != "l2" &&
      options.distance_function != "cosine") {
    usage_error("Distance function must be l1, l2, or cosine");
  }
  if (options.algorithm == "dijkstra") {
    if (options.points_path.empty())
      usage_error("Points path is required for dijkstra");
    if (options.distance_function.empty())
      usage_error("Distance function is required for dijkstra");
  }
  if (!std::isfinite(options.epsilon) || options.epsilon <= 0) {
    usage_error("Epsilon must be a finite positive number");
  }
  return options;
}

std::vector<std::filesystem::path> input_files(
    const std::filesystem::path &path) {
  std::vector<std::filesystem::path> files;
  if (std::filesystem::is_regular_file(path)) {
    files.push_back(path);
  } else if (std::filesystem::is_directory(path)) {
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
      if (entry.is_regular_file()) files.push_back(entry.path());
    }
  } else {
    throw std::runtime_error(
        "Input path does not exist or is not a file/directory: " +
        path.string());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    throw std::runtime_error("No input files found at " + path.string());
  }
  return files;
}

std::vector<std::string> split_words(const std::string &line) {
  std::istringstream       input(line);
  std::vector<std::string> words;
  std::string              word;
  while (input >> word) words.push_back(word);
  return words;
}

bool is_data_line(const std::string &line) {
  const auto first = line.find_first_not_of(" \t\r\n");
  return first != std::string::npos && line[first] != '#';
}

long long parse_integer(const std::string &value, const std::string &context) {
  std::size_t consumed = 0;
  long long   parsed;
  try {
    parsed = std::stoll(value, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error(context + ": invalid integer '" + value + "'");
  }
  if (consumed != value.size()) {
    throw std::runtime_error(context + ": invalid integer '" + value + "'");
  }
  return parsed;
}

point_id_t parse_point_id(const std::string &value,
                          const std::string &context) {
  const auto parsed = parse_integer(value, context);
  if (parsed < 0)
    throw std::runtime_error(context + ": point IDs must be non-negative");
  return static_cast<point_id_t>(parsed);
}

double parse_double(const std::string &value, const std::string &context) {
  std::size_t consumed = 0;
  double      parsed;
  try {
    parsed = std::stod(value, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error(context + ": invalid numeric value '" + value +
                             "'");
  }
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::runtime_error(context + ": expected a finite numeric value");
  }
  return parsed;
}

points_t read_points(const options_t &options) {
  std::map<point_id_t, std::vector<double>> points_by_id;
  std::size_t                               dimensions = 0;
  point_id_t                                next_id    = 0;
  const auto files = input_files(options.points_path);
  if (!options.points_have_ids && files.size() > 1) {
    throw std::runtime_error("Multiple point files require point IDs");
  }

  for (const auto &file : files) {
    std::ifstream input(file);
    std::cout << "Reading points file: " << file << std::endl;
    if (!input)
      throw std::runtime_error("Cannot open points file: " + file.string());
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      const auto words = split_words(line);
      if (words.empty()) continue;
      const auto  context = file.string() + ":" + std::to_string(line_number);
      std::size_t first_feature = 0;
      point_id_t  point_id      = next_id++;
      if (options.points_have_ids) {
        point_id      = parse_point_id(words.front(), context);
        first_feature = 1;
      }
      if (first_feature == words.size()) {
        throw std::runtime_error(context + ": feature vector is empty");
      }
      if (points_by_id.find(point_id) != points_by_id.end()) {
        throw std::runtime_error(context + ": duplicate point ID " +
                                 std::to_string(point_id));
      }
      std::vector<double> features;
      features.reserve(words.size() - first_feature);
      for (std::size_t index = first_feature; index < words.size(); ++index) {
        features.push_back(parse_double(words[index], context));
      }
      if (dimensions == 0) dimensions = features.size();
      if (features.size() != dimensions) {
        throw std::runtime_error(context + ": feature dimension mismatch");
      }
      points_by_id.emplace(point_id, std::move(features));
    }
  }
  if (points_by_id.empty())
    throw std::runtime_error("No feature vectors were loaded");
  const auto maximum_id = points_by_id.rbegin()->first;
  if (maximum_id == std::numeric_limits<point_id_t>::max()) {
    throw std::runtime_error("Point ID range is too large");
  }
  points_t points(maximum_id + 1, std::vector<double>(dimensions, 0.0));
  for (auto &[point_id, features] : points_by_id) {
    points[point_id] = std::move(features);
  }
  return points;
}

labels_t read_labels(const std::filesystem::path &path) {
  const auto          files = input_files(path);
  std::optional<bool> contains_ids;
  labels_t            labels;
  point_id_t          next_id = 0;
  for (const auto &file : files) {
    std::ifstream input(file);
    if (!input)
      throw std::runtime_error("Cannot open cluster file: " + file.string());
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      if (!is_data_line(line)) continue;
      const auto words = split_words(line);
      if (!contains_ids) {
        contains_ids = words.size() >= 2;
        if (files.size() > 1 && !*contains_ids) {
          throw std::runtime_error("Multiple cluster files require point IDs");
        }
      }
      const auto context = file.string() + ":" + std::to_string(line_number);
      const point_id_t point_id =
          *contains_ids ? parse_point_id(words[0], context) : next_id;
      const auto label_index = *contains_ids ? 1U : 0U;
      if (words.size() <= label_index) {
        throw std::runtime_error(context + ": expected a cluster label");
      }
      const auto cluster_id = parse_integer(words[label_index], context);
      if (!labels.emplace(point_id, cluster_id).second) {
        throw std::runtime_error(context + ": duplicate point ID " +
                                 std::to_string(point_id));
      }
      if (!*contains_ids) ++next_id;
    }
  }
  if (labels.empty()) throw std::runtime_error("No cluster labels were loaded");
  return labels;
}

graph_t read_graph(const options_t &options) {
  const auto files        = input_files(options.graph_path);
  const bool contains_ids = options.knng_format.back() == 'I';
  const char layout       = options.knng_format.front();
  graph_t    graph;
  point_id_t next_source_id = 0;

  for (const auto &file : files) {
    std::cout << "Reading graph file: " << file << std::endl;
    std::ifstream input(file);
    if (!input)
      throw std::runtime_error("Cannot open graph file: " + file.string());
    std::vector<std::pair<std::size_t, std::string>> lines;
    std::string                                      line;
    std::size_t                                      line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      if (is_data_line(line)) lines.emplace_back(line_number, line);
    }
    if (layout == 'B' && lines.size() % 2 != 0) {
      throw std::runtime_error(file.string() +
                               ": expected an even number of kNNG lines");
    }
    if (layout == 'M' && lines.size() % 2 != 0) {
      throw std::runtime_error(file.string() +
                               ": expected an even number of kNNG lines");
    }
    const auto row_count = lines.size() / 2;
    for (std::size_t row = 0; row < row_count; ++row) {
      const auto id_line_index = layout == 'B' ? row : row * 2;
      const auto distance_line_index =
          layout == 'B' ? row + row_count : row * 2 + 1;
      const auto context =
          file.string() + ":" + std::to_string(lines[id_line_index].first);
      const auto id_words = split_words(lines[id_line_index].second);
      if (contains_ids && id_words.empty()) {
        throw std::runtime_error(context + ": missing source point ID");
      }
      point_id_t  source_id;
      std::size_t first_neighbor = 0;
      if (contains_ids) {
        source_id      = parse_point_id(id_words[0], context);
        first_neighbor = 1;
      } else {
        source_id = next_source_id++;
      }
      std::vector<point_id_t> neighbors;
      for (std::size_t index = first_neighbor; index < id_words.size();
           ++index) {
        neighbors.push_back(parse_point_id(id_words[index], context));
      }
      const auto distance_context =
          file.string() + ":" +
          std::to_string(lines[distance_line_index].first);
      const auto distance_words =
          split_words(lines[distance_line_index].second);
      std::vector<double> distances;
      distances.reserve(distance_words.size());
      for (const auto &word : distance_words) {
        const auto distance = parse_double(word, distance_context);
        if (distance < 0) {
          throw std::runtime_error(distance_context +
                                   ": distances must be non-negative");
        }
        distances.push_back(distance);
      }
      const auto expected_distances =
          neighbors.size() + (contains_ids ? 1U : 0U);
      if (distances.size() != expected_distances) {
        throw std::runtime_error(distance_context +
                                 ": neighbor and distance counts do not match");
      }
      if (!graph.emplace(source_id, std::move(neighbors)).second) {
        throw std::runtime_error(context + ": duplicate source point ID " +
                                 std::to_string(source_id));
      }
    }
  }
  if (graph.empty()) throw std::runtime_error("No kNNG rows were loaded");
  return graph;
}

double distance(const std::vector<double> &first,
                const std::vector<double> &second,
                const std::string         &function) {
  if (first.size() != second.size() || first.empty()) {
    throw std::runtime_error(
        "Feature vectors must have matching non-empty dimensions");
  }
  double value = 0.0;
  if (function == "l1") {
    for (std::size_t index = 0; index < first.size(); ++index) {
      value += std::abs(first[index] - second[index]);
    }
  } else if (function == "l2") {
    for (std::size_t index = 0; index < first.size(); ++index) {
      const double difference = first[index] - second[index];
      value += difference * difference;
    }
    value = std::sqrt(value);
  } else {
    double dot         = 0.0;
    double norm_first  = 0.0;
    double norm_second = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
      dot += first[index] * second[index];
      norm_first += first[index] * first[index];
      norm_second += second[index] * second[index];
    }
    if (norm_first == 0.0 || norm_second == 0.0) {
      throw std::runtime_error("Cosine distance is undefined for zero vectors");
    }
    const double similarity =
        std::clamp(dot / std::sqrt(norm_first * norm_second), -1.0, 1.0);
    value = 1.0 - similarity;
  }
  if (!std::isfinite(value) || value < 0.0) {
    throw std::runtime_error("Distance function returned an invalid value");
  }
  return value;
}

const std::vector<point_id_t> &neighbors_of(const graph_t &graph,
                                            point_id_t     point_id) {
  static const std::vector<point_id_t> empty;
  const auto                           found = graph.find(point_id);
  return found == graph.end() ? empty : found->second;
}

void validate_point_id(point_id_t point_id, const points_t &points) {
  if (point_id >= points.size()) {
    throw std::runtime_error(
        "Graph point ID is outside the loaded feature vectors: " +
        std::to_string(point_id));
  }
}

std::pair<std::optional<match_t>, std::optional<match_t>> find_bfs(
    point_id_t source_id, cluster_id_t source_cluster, const labels_t &labels,
    const graph_t &graph) {
  std::optional<match_t>         closest_in;
  std::optional<match_t>         closest_out;
  std::unordered_set<point_id_t> visited{source_id};
  std::vector<point_id_t>        frontier{source_id};
  std::size_t                    hops = 0;

  while (!frontier.empty() && (!closest_in || !closest_out)) {
    std::vector<point_id_t> next_frontier;
    std::optional<match_t>  layer_in;
    std::optional<match_t>  layer_out;
    for (const auto point_id : frontier) {
      if (point_id != source_id) {
        const auto label = labels.find(point_id);
        if (label != labels.end() && label->second >= 0 &&
            ((!closest_in && label->second == source_cluster) ||
             (!closest_out && label->second != source_cluster))) {
          auto &layer_match =
              label->second == source_cluster ? layer_in : layer_out;
          if (!layer_match || point_id < layer_match->point_id) {
            layer_match = match_t{point_id, hops, static_cast<double>(hops)};
          }
        }
      }
      for (const auto neighbor_id : neighbors_of(graph, point_id)) {
        if (visited.insert(neighbor_id).second)
          next_frontier.push_back(neighbor_id);
      }
    }
    if (!closest_in && layer_in) closest_in = layer_in;
    if (!closest_out && layer_out) closest_out = layer_out;
    frontier = std::move(next_frontier);
    ++hops;
  }
  return {closest_in, closest_out};
}

std::pair<std::optional<match_t>, std::optional<match_t>> find_dijkstra(
    point_id_t source_id, cluster_id_t source_cluster, const labels_t &labels,
    const graph_t &graph, const points_t &points,
    const std::string &distance_function) {
  using frontier_item_t = std::tuple<double, std::size_t, point_id_t>;
  std::priority_queue<frontier_item_t, std::vector<frontier_item_t>,
                      std::greater<frontier_item_t>>
      frontier;
  frontier.emplace(0.0, 0, source_id);
  std::unordered_set<point_id_t> visited{source_id};
  std::optional<match_t>         closest_in;
  std::optional<match_t>         closest_out;

  while (!frontier.empty()) {
    const auto [source_distance, hops, point_id] = frontier.top();
    (void)source_distance;
    frontier.pop();
    for (const auto neighbor_id : neighbors_of(graph, point_id)) {
      if (!visited.insert(neighbor_id).second) continue;
      validate_point_id(neighbor_id, points);
      const auto neighbor_distance =
          distance(points[source_id], points[neighbor_id], distance_function);
      const auto label = labels.find(neighbor_id);
      if (label != labels.end() && label->second >= 0) {
        auto &match =
            label->second == source_cluster ? closest_in : closest_out;
        if (!match || neighbor_distance < match->distance) {
          match = match_t{neighbor_id, hops + 1, neighbor_distance};
        }
      }
      frontier.emplace(neighbor_distance, hops + 1, neighbor_id);
    }
  }
  return {closest_in, closest_out};
}

void make_undirected(graph_t &graph) {
  for (const auto &[point_id, neighbors] : graph) {
    for (const auto neighbor_id : neighbors) {
      graph[neighbor_id].push_back(point_id);
    }
  }

  // Remove duplicate neighbors
  for (auto &[point_id, neighbors] : graph) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }
}

std::vector<result_t> find_nearest_in_and_out_cluster_points(
    const graph_t &graph, const std::optional<points_t> &points,
    const labels_t &all_labels, const options_t &options) {
  if (options.algorithm == "dijkstra" && !points) {
    throw std::invalid_argument("Dijkstra requires feature vectors");
  }
  std::vector<std::pair<point_id_t, cluster_id_t>> ordered_labels;
  ordered_labels.reserve(all_labels.size());
  for (const auto &[point_id, cluster_id] : all_labels) {
    if (cluster_id >= 0) ordered_labels.emplace_back(point_id, cluster_id);
  }
  std::sort(ordered_labels.begin(), ordered_labels.end());

  std::vector<result_t>           results(ordered_labels.size());
  std::vector<std::exception_ptr> errors(ordered_labels.size());
#pragma omp parallel for
  for (std::ptrdiff_t index = 0;
       index < static_cast<std::ptrdiff_t>(ordered_labels.size()); ++index) {
    try {
      const auto [source_id, source_cluster] =
          ordered_labels[static_cast<std::size_t>(index)];
      std::pair<std::optional<match_t>, std::optional<match_t>> matches;
      if (options.algorithm == "bfs") {
        matches = find_bfs(source_id, source_cluster, all_labels, graph);
      } else {
        validate_point_id(source_id, *points);
        matches = find_dijkstra(source_id, source_cluster, all_labels, graph,
                                *points, options.distance_function);
      }
      results[static_cast<std::size_t>(index)] = {source_id, source_cluster,
                                                  std::move(matches.first),
                                                  std::move(matches.second)};
    } catch (...) {
      errors[static_cast<std::size_t>(index)] = std::current_exception();
    }
  }
  for (const auto &error : errors) {
    if (error) std::rethrow_exception(error);
  }
  return results;
}

void write_per_point_score(const std::filesystem::path &output_path,
                           const std::vector<result_t> &results) {
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output)
    throw std::runtime_error("Cannot open output file: " +
                             output_path.string());
  output << "point_id\tcluster_id\tclosest_in_point_id\tclosest_in_hops\t"
            "closest_in_distance\tclosest_out_point_id\tclosest_out_hops\t"
            "closest_out_distance\n";
  output << std::setprecision(17);
  for (const auto &result : results) {
    output << result.point_id << '\t' << result.cluster_id << '\t';
    if (result.closest_in) {
      output << result.closest_in->point_id << '\t' << result.closest_in->hops
             << '\t' << result.closest_in->distance;
    } else {
      output << "None\tNone\tNone";
    }
    output << '\t';
    if (result.closest_out) {
      output << result.closest_out->point_id << '\t' << result.closest_out->hops
             << '\t' << result.closest_out->distance;
    } else {
      output << "None\tNone\tNone";
    }
    output << '\n';
  }
  if (!output)
    throw std::runtime_error("Failed writing output file: " +
                             output_path.string());
  std::cout << "Per-point results written to " << output_path << '\n';
}

struct cluster_score_t {
  size_t valid_points{0};
  double score{0.0};
  size_t invalid_points{0};
};

std::unordered_map<cluster_id_t, cluster_score_t> compute_cluster_score(
    const std::vector<result_t> &results) {
  std::unordered_map<cluster_id_t, cluster_score_t> cluster_scores;
  for (const auto &result : results) {
    const auto cluster_id = result.cluster_id;
    if (!result.closest_in || !result.closest_out) {
      ++cluster_scores[cluster_id].invalid_points;
    } else {
      const auto in_dist  = result.closest_in->distance;
      const auto out_dist = result.closest_out->distance;
      cluster_scores[cluster_id].score += in_dist / out_dist;
      ++cluster_scores[cluster_id].valid_points;
    }
  }

  for (auto &entry : cluster_scores) {
    if (entry.second.valid_points > 0) {
      entry.second.score /= entry.second.valid_points;
    }
  }

  return cluster_scores;
}

void print_cluster_scores(
    std::ostream                                            &output,
    const std::unordered_map<cluster_id_t, cluster_score_t> &cluster_scores,
    const size_t max_n = std::numeric_limits<size_t>::max()) {
  std::vector<cluster_id_t> cluster_ids;
  cluster_ids.reserve(cluster_scores.size());
  for (const auto &entry : cluster_scores) {
    cluster_ids.push_back(entry.first);
  }
  std::sort(cluster_ids.begin(), cluster_ids.end());

  output << "ClusterID\tValidPoints\tInvalidPoints\tScore\n";
  size_t count = 0;
  for (const auto cluster_id : cluster_ids) {
    if (count >= max_n) break;
    const auto &score = cluster_scores.at(cluster_id);
    output << cluster_id << '\t' << score.valid_points << '\t'
           << score.invalid_points << '\t' << score.score << std::endl;
    ;
    ++count;
  }
}

void write_cluster_scores(
    const std::filesystem::path                             &output_path,
    const std::unordered_map<cluster_id_t, cluster_score_t> &cluster_scores) {
  std::ofstream output(output_path);
  if (!output)
    throw std::runtime_error("Failed opening output file: " +
                             output_path.string());
  print_cluster_scores(output, cluster_scores);
  if (!output)
    throw std::runtime_error("Failed writing output file: " +
                             output_path.string());
  std::cout << "Cluster scores written to " << output_path << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const auto              options = parse_options(argc, argv);
    std::optional<points_t> points;
    if (options.algorithm == "dijkstra") {
      std::cout << "Reading points.." << std::endl;
      points = read_points(options);
      std::cout << "Loaded " << points->size() << " point" << std::endl;
    }
    std::cout << "Reading graph.." << std::endl;
    auto graph = read_graph(options);
    std::cout << "Loaded " << graph.size() << " graph point" << std::endl;
    std::cout << "Reading cluster labels.." << std::endl;
    const auto labels = read_labels(options.cluster_path);
    const auto clustered_count =
        std::count_if(labels.begin(), labels.end(),
                      [](const auto &entry) { return entry.second >= 0; });
    std::cout << "Loaded " << clustered_count << " assigned cluster label"
              << std::endl;

    std::cout << "Making graph undirected.." << std::endl;
    make_undirected(graph);
    std::cout << "Graph made undirected." << std::endl;

    const auto results =
        find_nearest_in_and_out_cluster_points(graph, points, labels, options);
    std::cout << "Found nearest in and out cluster points." << std::endl;

    if (!options.output_path.empty()) {
      write_per_point_score(options.output_path.string() + "-per_point.txt",
                            results);
    }

    const auto cluster_scores = compute_cluster_score(results);
    if (!options.output_path.empty()) {
      write_cluster_scores(options.output_path.string() + "-cluster.txt",
                           cluster_scores);
    } else {
      print_cluster_scores(std::cout, cluster_scores, 100);
    }

  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
