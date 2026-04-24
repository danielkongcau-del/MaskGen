#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/partition_2.h>
#include <CGAL/version.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Traits = CGAL::Partition_traits_2<Kernel>;
using Point = Traits::Point_2;
using PartitionPolygon = Traits::Polygon_2;
using CheckPolygon = CGAL::Polygon_2<Kernel>;

std::string read_text_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    output << text;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
}

std::string error_payload(const std::string& error) {
    std::ostringstream out;
    out << "{\n"
        << "  \"success\": false,\n"
        << "  \"error\": \"" << json_escape(error) << "\"\n"
        << "}\n";
    return out.str();
}

std::string extract_outer_array(const std::string& json_text) {
    const std::size_t key_pos = json_text.find("\"outer\"");
    if (key_pos == std::string::npos) {
        throw std::runtime_error("input JSON does not contain an outer field");
    }
    const std::size_t start = json_text.find('[', key_pos);
    if (start == std::string::npos) {
        throw std::runtime_error("outer field is not an array");
    }

    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t index = start; index < json_text.size(); ++index) {
        const char ch = json_text[index];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
        } else if (ch == '[') {
            ++depth;
        } else if (ch == ']') {
            --depth;
            if (depth == 0) {
                return json_text.substr(start, index - start + 1);
            }
        }
    }
    throw std::runtime_error("outer array is not closed");
}

std::vector<Point> parse_outer_points(const std::string& json_text) {
    const std::string outer_array = extract_outer_array(json_text);
    const std::regex number_regex(R"([-+]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][-+]?\d+)?)");

    std::vector<double> values;
    for (auto it = std::sregex_iterator(outer_array.begin(), outer_array.end(), number_regex);
         it != std::sregex_iterator();
         ++it) {
        values.push_back(std::stod(it->str()));
    }
    if (values.size() < 6 || values.size() % 2 != 0) {
        throw std::runtime_error("outer field must contain at least three [x, y] coordinate pairs");
    }

    std::vector<Point> points;
    points.reserve(values.size() / 2);
    for (std::size_t index = 0; index < values.size(); index += 2) {
        points.emplace_back(values[index], values[index + 1]);
    }

    if (points.size() > 1 && points.front() == points.back()) {
        points.pop_back();
    }
    if (points.size() < 3) {
        throw std::runtime_error("outer ring has fewer than three unique vertices");
    }
    return points;
}

double to_double(const Kernel::FT& value) {
    return CGAL::to_double(value);
}

std::string success_payload(
    const std::vector<Point>& input_points,
    const std::list<PartitionPolygon>& pieces,
    bool cgal_valid) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"success\": true,\n";
    out << "  \"backend_info\": {\n";
    out << "    \"backend\": \"cgal\",\n";
    out << "    \"optimal\": true,\n";
    out << "    \"algorithm\": \"CGAL::optimal_convex_partition_2\",\n";
    out << "    \"cgal_version\": \"" << CGAL_VERSION_STR << "\",\n";
    out << "    \"input_vertex_count\": " << input_points.size() << ",\n";
    out << "    \"cgal_valid\": " << (cgal_valid ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"piece_count\": " << pieces.size() << ",\n";
    out << "  \"pieces\": [\n";

    std::size_t piece_index = 0;
    for (const auto& piece : pieces) {
        out << "    [";
        for (auto vertex = piece.vertices_begin(); vertex != piece.vertices_end(); ++vertex) {
            if (vertex != piece.vertices_begin()) {
                out << ", ";
            }
            out << "[" << to_double(vertex->x()) << ", " << to_double(vertex->y()) << "]";
        }
        out << "]";
        if (++piece_index < pieces.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

std::string run_partition(const std::string& input_json) {
    std::vector<Point> points = parse_outer_points(input_json);

    CheckPolygon check_polygon(points.begin(), points.end());
    if (!check_polygon.is_simple()) {
        throw std::runtime_error("input polygon is not simple");
    }
    const auto orientation = check_polygon.orientation();
    if (orientation == CGAL::COLLINEAR) {
        throw std::runtime_error("input polygon is degenerate or collinear");
    }
    if (orientation == CGAL::CLOCKWISE) {
        std::reverse(points.begin(), points.end());
    }

    Traits traits;
    std::list<PartitionPolygon> pieces;
    CGAL::optimal_convex_partition_2(
        points.begin(), points.end(), std::back_inserter(pieces), traits);
    const bool cgal_valid = CGAL::convex_partition_is_valid_2(
        points.begin(), points.end(), pieces.begin(), pieces.end(), traits);
    if (!cgal_valid) {
        throw std::runtime_error("CGAL convex partition validation failed");
    }
    return success_payload(points, pieces, cgal_valid);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: optimal_convex_partition_cli <input.json> <output.json>\n";
        return 2;
    }

    try {
        const std::string input_json = read_text_file(argv[1]);
        write_text_file(argv[2], run_partition(input_json));
        return 0;
    } catch (const std::exception& error) {
        try {
            write_text_file(argv[2], error_payload(error.what()));
        } catch (...) {
            std::cerr << error.what() << "\n";
            return 1;
        }
        return 0;
    }
}
