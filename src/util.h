#pragma once

#include "TriMesh.h"
#include <glm/glm.hpp>
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdint.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>

template<typename trimeshtype>
inline glm::vec3 trimesh_to_glm(trimeshtype a) {
	return glm::vec3(a[0], a[1], a[2]);
}

inline char checkVoxel(size_t x, size_t y, size_t z, size_t gridsize, const unsigned int* vtable){
	size_t location = x + (y*gridsize) + (z*gridsize*gridsize);
	size_t int_location = location / size_t(32);
	/*size_t max_index = (gridsize*gridsize*gridsize) / __int64(32);
	if (int_location >= max_index){
	fprintf(stdout, "Requested index too big: %llu \n", int_location);
	fprintf(stdout, "X %llu Y %llu Z %llu \n", int_location);
	}*/
	unsigned int bit_pos = size_t(31) - (location % size_t(32)); // we count bit positions RtL, but array indices LtR
	if ((vtable[int_location]) & (1 << bit_pos)){
		return char(1);
	}
	return char(0);
}

template <typename T>
struct AABox {
	T min;
	T max;
	__device__ __host__ AABox() : min(T()), max(T()) {}
	__device__ __host__ AABox(T min, T max) : min(min), max(max) {}
};

// voxelisation info (same for every triangle)
struct voxinfo {
	AABox<glm::vec3> bbox;
	glm::uvec3 gridsize;
	size_t n_triangles;
	glm::vec3 unit;

	voxinfo(AABox<glm::vec3> bbox, glm::uvec3 gridsize, size_t n_triangles)
		: gridsize(gridsize), bbox(bbox), n_triangles(n_triangles) {
		unit.x = (bbox.max.x - bbox.min.x) / float(gridsize.x);
		unit.y = (bbox.max.y - bbox.min.y) / float(gridsize.y);
		unit.z = (bbox.max.z - bbox.min.z) / float(gridsize.z);
	}

	void print() {
		fprintf(stdout, "Bounding Box: (%f, %f, %f) to (%f, %f, %f) \n", bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
		fprintf(stdout, "Grid size: %i \n", gridsize);
		fprintf(stdout, "Triangles: %ull \n", n_triangles);
		fprintf(stdout, "Unit length: x: %f y: %f z: %f\n", unit.x, unit.y, unit.z);
	}
};

// create mesh bbox cube
template <typename T>
__device__ __host__ inline AABox<T> createMeshBBCube(AABox<T> box) {
	AABox<T> answer(box.min, box.max);
	glm::vec3 lengths = box.max - box.min;
	float max_length = glm::max(lengths.x, glm::max(lengths.y, lengths.z));
	for (int i = 0; i < 3; i++) {
		float delta = max_length - lengths[i];
		if (delta != 0) {
			answer.min[i] = box.min[i] - (delta / 2.0f);
			answer.max[i] = box.max[i] + (delta / 2.0f);
		}
	}
	return answer;
}



__host__ __device__ void inline printBits(size_t const size, void const * const ptr) {
	unsigned char *b = (unsigned char*)ptr;
	unsigned char byte;
	int i, j;
	for (i = size - 1; i >= 0; i--) {
		for (j = 7; j >= 0; j--) {
			byte = b[i] & (1 << j);
			byte >>= j;
			if (byte) {
				printf("X");
			}
			else {
				printf(".");
			}
			//printf("%u", byte);
		}
	}
	puts("");
}
