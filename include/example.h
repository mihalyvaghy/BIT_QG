#include "qgedge.h"
#include <fstream>
#include <vector>
#include <unordered_set>
#include <Eigen/Core>

#pragma once

#define PI 3.14159265358979323846

class Example {
public:
	std::vector<QGEdge> edges;
	int vertices;
	std::vector<std::unordered_set<int>> neighbours;

	Example() {}
	Example(std::vector<QGEdge> _edges, int _vertices) : edges(_edges), vertices(_vertices) {}
	Example(std::string filename) {
		int nnz = 0;
		int row = 0;
		int col;
		int num;
		std::ifstream f("../graphs/"+filename+".txt");
		std::string line;
		while (getline(f, line)) {
			col = 0;

			std::istringstream iss(line);
			while (iss >> num) {
				if (num) {
					edges.emplace_back(row, col,
						[](double x){return 1.0/(1.0+exp(-25*(x-0.5)))+1.0;},
						[](double x){return 0.05/pow(0.2,2)*pow(abs(x-0.5)-0.2,2)+0.05;},
						[](double x){return 1.0*exp(-(x-0.0)*(x-0.0)*250*4);});
					++nnz;
				}
				++col;

				if (iss.peek() == ',')
					iss.ignore();
			}
			++row;
		}
		vertices = row;
		neighbours.resize(vertices);
		for (QGEdge& edge : edges) {
			neighbours[edge.in].emplace(edge.out);
			neighbours[edge.out].emplace(edge.in);
		}
	}
};
