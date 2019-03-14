#include "voxelize.cuh"

// CUDA Global Memory variables
//__device__ size_t voxel_count = 0; // How many voxels did we count
//__device__ size_t triangles_seen_count = 0; // Sanity check

__constant__ uint32_t morton256_x[256];
__constant__ uint32_t morton256_y[256];
__constant__ uint32_t morton256_z[256];

// Encode morton code using LUT table
__device__ inline uint64_t mortonEncode_LUT(unsigned int x, unsigned int y, unsigned int z){
	uint64_t answer = 0;
	answer = morton256_z[(z >> 16) & 0xFF] |
		morton256_y[(y >> 16) & 0xFF] |
		morton256_x[(x >> 16) & 0xFF];
	answer = answer << 48 |
		morton256_z[(z >> 8) & 0xFF] |
		morton256_y[(y >> 8) & 0xFF] |
		morton256_x[(x >> 8) & 0xFF];
	answer = answer << 24 |
		morton256_z[(z)& 0xFF] |
		morton256_y[(y)& 0xFF] |
		morton256_x[(x)& 0xFF];
	return answer;
}

// Possible optimization: buffer bitsets (for now: Disabled because too much overhead)
//struct bufferedBitSetter{
//	unsigned int* voxel_table;
//	size_t current_int_location;
//	unsigned int current_mask;
//
//	__device__ __inline__ bufferedBitSetter(unsigned int* voxel_table, size_t index) :
//		voxel_table(voxel_table), current_mask(0) {
//		current_int_location = int(index / 32.0f);
//	}
//
//	__device__ __inline__ void setBit(size_t index){
//		size_t new_int_location = int(index / 32.0f);
//		if (current_int_location != new_int_location){
//			flush();
//			current_int_location = new_int_location;
//		}
//		unsigned int bit_pos = 31 - (unsigned int)(int(index) % 32);
//		current_mask = current_mask | (1 << bit_pos);
//	}
//
//	__device__ __inline__ void flush(){
//		if (current_mask != 0){
//			atomicOr(&(voxel_table[current_int_location]), current_mask);
//		}
//	}
//};

// Possible optimization: check bit before you set it - don't need to do atomic operation if it's already set to 1
// For now: overhead, so it seems
__device__ __inline__ bool checkBit(unsigned int* voxel_table, size_t index){
	size_t int_location = index / size_t(32);
	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
	return ((voxel_table[int_location]) & (1 << bit_pos));
}
__device__ __inline__ void flipBit(unsigned int* voxel_table, size_t index) {
  size_t int_location = index / size_t(32);
  unsigned int bit_pos = size_t(31) - (index % size_t(32));
  unsigned int mask;
  mask = 1 << bit_pos;
  atomicXor(&(voxel_table[int_location]), mask);
}

// Set a bit in the giant voxel table. This involves doing an atomic operation on a 32-bit word in memory.
// Blocking other threads writing to it for a very short time
__device__ __inline__ void setBit(unsigned int* voxel_table, size_t index){
	size_t int_location = index / size_t(32);
	unsigned int bit_pos = size_t(31) - (index % size_t(32)); // we count bit positions RtL, but array indices LtR
	unsigned int mask = 1 << bit_pos;
	atomicOr(&(voxel_table[int_location]), mask);
}

__device__ __inline__ void unsetBit(unsigned int* voxel_table, size_t index) {

  size_t int_location = index / size_t(32);
  unsigned int bit_pos = size_t(31) - (index & size_t(32));
  unsigned int mask = 1 << bit_pos;
  atomicAnd(&(voxel_table[int_location]), ~mask);
}


// Main triangle voxelization method
__global__ void voxelize_triangle(voxinfo info, float* triangle_data, unsigned int* voxel_table, bool morton_order){
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	// Common variables used in the voxelization process
	glm::vec3 delta_p(info.unit.x, info.unit.y, info.unit.z);
	glm::vec3 c(0.0f, 0.0f, 0.0f); // critical point
	glm::vec3 grid_max(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)

	while (thread_id < info.n_triangles){ // every thread works on specific triangles in its stride
		size_t t = thread_id * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using bbox
		glm::vec3 v0 = glm::vec3(triangle_data[t], triangle_data[t + 1], triangle_data[t + 2]) - info.bbox.min;
		glm::vec3 v1 = glm::vec3(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info.bbox.min; 
		glm::vec3 v2 = glm::vec3(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info.bbox.min;
		// Edge vectors
		glm::vec3 e0 = v1 - v0;
		glm::vec3 e1 = v2 - v1;
		glm::vec3 e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		glm::vec3 n = glm::normalize(glm::cross(e0, e1));

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		AABox<glm::vec3> t_bbox_world(glm::min(v0, glm::min(v1, v2)), glm::max(v0, glm::max(v1, v2)));
		// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<glm::ivec3> t_bbox_grid;
		t_bbox_grid.min = glm::clamp(t_bbox_world.min / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);
		t_bbox_grid.max = glm::clamp(t_bbox_world.max / info.unit, glm::vec3(0.0f, 0.0f, 0.0f), grid_max);

		// PREPARE PLANE TEST PROPERTIES
		if (n.x > 0.0f) { c.x = info.unit.x; }
		if (n.y > 0.0f) { c.y = info.unit.y; }
		if (n.z > 0.0f) { c.z = info.unit.z; }
		float d1 = glm::dot(n, (c - v0));
		float d2 = glm::dot(n, ((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		glm::vec2 n_xy_e0(-1.0f*e0.y, e0.x);
		glm::vec2 n_xy_e1(-1.0f*e1.y, e1.x);
		glm::vec2 n_xy_e2(-1.0f*e2.y, e2.x);
		if (n.z < 0.0f) {
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		float d_xy_e0 = (-1.0f * glm::dot(n_xy_e0, glm::vec2(v0.x, v0.y))) + glm::max(0.0f, info.unit.x*n_xy_e0[0]) + glm::max(0.0f, info.unit.y*n_xy_e0[1]);
		float d_xy_e1 = (-1.0f * glm::dot(n_xy_e1, glm::vec2(v1.x, v1.y))) + glm::max(0.0f, info.unit.x*n_xy_e1[0]) + glm::max(0.0f, info.unit.y*n_xy_e1[1]);
		float d_xy_e2 = (-1.0f * glm::dot(n_xy_e2, glm::vec2(v2.x, v2.y))) + glm::max(0.0f, info.unit.x*n_xy_e2[0]) + glm::max(0.0f, info.unit.y*n_xy_e2[1]);
		// YZ plane
		glm::vec2 n_yz_e0(-1.0f*e0.z, e0.y);
		glm::vec2 n_yz_e1(-1.0f*e1.z, e1.y);
		glm::vec2 n_yz_e2(-1.0f*e2.z, e2.y);
		if (n.x < 0.0f) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		float d_yz_e0 = (-1.0f * glm::dot(n_yz_e0, glm::vec2(v0.y, v0.z))) + glm::max(0.0f, info.unit.y*n_yz_e0[0]) + glm::max(0.0f, info.unit.z*n_yz_e0[1]);
		float d_yz_e1 = (-1.0f * glm::dot(n_yz_e1, glm::vec2(v1.y, v1.z))) + glm::max(0.0f, info.unit.y*n_yz_e1[0]) + glm::max(0.0f, info.unit.z*n_yz_e1[1]);
		float d_yz_e2 = (-1.0f * glm::dot(n_yz_e2, glm::vec2(v2.y, v2.z))) + glm::max(0.0f, info.unit.y*n_yz_e2[0]) + glm::max(0.0f, info.unit.z*n_yz_e2[1]);
		// ZX plane
		glm::vec2 n_zx_e0(-1.0f*e0.x, e0.z);
		glm::vec2 n_zx_e1(-1.0f*e1.x, e1.z);
		glm::vec2 n_zx_e2(-1.0f*e2.x, e2.z);
		if (n.y < 0.0f) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * glm::dot(n_zx_e0, glm::vec2(v0.z, v0.x))) + glm::max(0.0f, info.unit.x*n_zx_e0[0]) + glm::max(0.0f, info.unit.z*n_zx_e0[1]);
		float d_xz_e1 = (-1.0f * glm::dot(n_zx_e1, glm::vec2(v1.z, v1.x))) + glm::max(0.0f, info.unit.x*n_zx_e1[0]) + glm::max(0.0f, info.unit.z*n_zx_e1[1]);
		float d_xz_e2 = (-1.0f * glm::dot(n_zx_e2, glm::vec2(v2.z, v2.x))) + glm::max(0.0f, info.unit.x*n_zx_e2[0]) + glm::max(0.0f, info.unit.z*n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; z++){
			for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++){
				for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++){
					// size_t location = x + (y*info.gridsize) + (z*info.gridsize*info.gridsize);
					// if (checkBit(voxel_table, location)){ continue; }

					// TRIANGLE PLANE THROUGH BOX TEST
					glm::vec3 p(x*info.unit.x, y*info.unit.y, z*info.unit.z);
					float nDOTp = glm::dot(n, p);
					if ((nDOTp + d1) * (nDOTp + d2) > 0.0f){ continue; }

					// PROJECTION TESTS
					// XY
					glm::vec2 p_xy(p.x, p.y);
					if ((glm::dot(n_xy_e0, p_xy) + d_xy_e0) < 0.0f){ continue; }
					if ((glm::dot(n_xy_e1, p_xy) + d_xy_e1) < 0.0f){ continue; }
					if ((glm::dot(n_xy_e2, p_xy) + d_xy_e2) < 0.0f){ continue; }

					// YZ
					glm::vec2 p_yz(p.y, p.z);
					if ((glm::dot(n_yz_e0, p_yz) + d_yz_e0) < 0.0f){ continue; }
					if ((glm::dot(n_yz_e1, p_yz) + d_yz_e1) < 0.0f){ continue; }
					if ((glm::dot(n_yz_e2, p_yz) + d_yz_e2) < 0.0f){ continue; }

					// XZ	
					glm::vec2 p_zx(p.z, p.x);
					if ((glm::dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f){ continue; }

					//atomicAdd(&voxel_count, 1);
					if (morton_order){
						size_t location = mortonEncode_LUT(x, y, z);
						setBit(voxel_table, location);
					} else {
						size_t location = x + (y*info.gridsize.y) + (z*info.gridsize.y*info.gridsize.z);
						setBit(voxel_table, location);
					}
					continue;
				}
			}
		}
		// sanity check: atomically count triangles
		//atomicAdd(&triangles_seen_count, 1);
		thread_id += stride;
	}
}


__global__ void createPlane(voxinfo info, unsigned int* vtable) {
  int x = threadIdx.x;
  int z = blockIdx.x;

  size_t location = x + (10 * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
  setBit(vtable, location);
  location = x + (30 * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
  setBit(vtable, location);
}


__global__ void rasterization(voxinfo info, float* triangle_data, unsigned int* voxel_table, bool morton_order) {
  size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;

  glm::vec3 delta_p(info.unit.x, info.unit.y, info.unit.z);
  glm::vec3 grid_max(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1);
  
  while (thread_id < info.n_triangles) {
    size_t t = thread_id * 9;

    glm::vec3 v0 = glm::vec3(triangle_data[t    ], triangle_data[t + 1], triangle_data[t + 2]) - info.bbox.min;
    glm::vec3 v1 = glm::vec3(triangle_data[t + 3], triangle_data[t + 4], triangle_data[t + 5]) - info.bbox.min;
    glm::vec3 v2 = glm::vec3(triangle_data[t + 6], triangle_data[t + 7], triangle_data[t + 8]) - info.bbox.min;

    glm::vec3 e0 = v1 - v0;
    glm::vec3 e1 = v2 - v1;
    glm::vec3 e2 = v0 - v2;

    glm::vec3 normal = glm::normalize(glm::cross(e0,e1));


    // COMPUTE TRIANGLE BOZ IN GRID
    AABox<glm::vec3> t_bbox_world(glm::min(v0, glm::min(v1, v2)), glm::max(v0, glm::max(v1, v2)));
    AABox<glm::ivec3> t_bbox_grid;
    t_bbox_grid.min = glm::clamp(t_bbox_world.min / info.unit, glm::vec3(0.0f), grid_max);
    t_bbox_grid.max = glm::clamp(t_bbox_world.max / info.unit, glm::vec3(0.0f), grid_max);

    // zx plane checks
		glm::vec2 n_zx_e0(-1.0f*e0.x, e0.z);
		glm::vec2 n_zx_e1(-1.0f*e1.x, e1.z);
		glm::vec2 n_zx_e2(-1.0f*e2.x, e2.z);
		if (normal.y < 0.0f) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * glm::dot(n_zx_e0, glm::vec2(v0.z, v0.x)));
		float d_xz_e1 = (-1.0f * glm::dot(n_zx_e1, glm::vec2(v1.z, v1.x)));
		float d_xz_e2 = (-1.0f * glm::dot(n_zx_e2, glm::vec2(v2.z, v2.x)));

    for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; ++z) {
      for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; ++x) {
        for (int y = t_bbox_grid.min.y; y <= grid_max.y; ++y) {
          glm::vec3 voxel_centre(x * info.unit.x + info.unit.x / 2.0f,
                                 y * info.unit.y + info.unit.y / 2.0f,
                                 z * info.unit.z + info.unit.z / 2.0f);
          
					// XZ	
					glm::vec2 p_zx(voxel_centre.z, voxel_centre.x);
					if ((glm::dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f){ continue; }
					if ((glm::dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f){ continue; }

          size_t location; 
          if (morton_order) {
            location = mortonEncode_LUT(x, y, z);
          } else {
            location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
          }

          if (checkVoxel(x, y - 1, z, info.gridsize.x, voxel_table) == char(1)) 
            continue;

          setBit(voxel_table, location);
          break;
        }
      }
    }

  thread_id += stride;
  }
}

//__global__ void fill_pass(voxinfo info, float* triangle_data, unsigned int* vtable, bool morton_order) {
//  int x = threadIdx.x;
//  int z = blockIdx.x;
//
//  int ymax = info.gridsize.y - 1;
//
//  bool flip = false;
//  for (int y = 1; y < ymax; ++y) {
//    char value = checkVoxel(x, y, z, info.gridsize.x, vtable);
//     flip = true;
//
//    
//    size_t location;
//    if (morton_order) {
//      location = mortonEncode_LUT(x, y, z);
//    } else {
//      location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
//    }
//
//    size_t int_location = location / size_t(32);
//    unsigned int bit_pos = size_t(31) - (location % size_t(32));
//
//    unsigned int mask = 0;
//    if (prev_value == char(1)) {
//      mask = 1 << bit_pos;
//    } 
//    atomicXor(&(vtable[int_location]), mask);
//  }
//}


__global__ void carveY(unsigned int* vtable_obj, unsigned int* vtable_carved, 
                       const voxinfo& info, bool morton_order) {
  int x = threadIdx.x;
  int z = blockIdx.x;

  size_t ymax = info.gridsize.y - 1;

  for (size_t y = 0; y < ymax; ++y) {
    size_t location;
    if (morton_order) {
      location = mortonEncode_LUT(x, y, z);
    } else {
      location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
    }

    if (checkBit(vtable_obj, location)) { 
      break;
    }

    unsetBit(vtable_carved, location);
  }

  for (size_t y = ymax; y >= 0; --y) {
    size_t location;
    if (morton_order) {
      location = mortonEncode_LUT(x, y, z);
    } else {
      location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
    }

    if (checkBit(vtable_obj, location)) {
      break;
    }


    unsetBit(vtable_carved, location);
  }
}


__global__ void carveX(unsigned int* vtable_obj, unsigned int* vtable_carved, 
                      const voxinfo& info, bool morton_order) {
  int y = threadIdx.x;
  int z = blockIdx.x;

  size_t xmax = info.gridsize.x - 1;
  
  for (size_t x = 0; x <= xmax; ++x) {
    size_t location;

    if (morton_order) {
      location = mortonEncode_LUT(x, y, z);
    } else {
      location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
    }

    if (checkBit(vtable_obj, location)) {
      break;
    }


    unsetBit(vtable_carved, location);
  }

  for (size_t x = xmax; x >= 0; --x) {

    size_t location;

    if (morton_order) {
      location = mortonEncode_LUT(x, y ,z);
    } else {
      location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
    }

    if (checkBit(vtable_obj, location)) {
      break;
    }

    unsetBit(vtable_carved, location);
  }
}


__global__ void carveZ(unsigned int* vtable_obj, unsigned int* vtable_carved, 
                       const voxinfo& info, bool morton_order) {
  int x = threadIdx.x;
  int y = blockIdx.x;

  size_t zmax = info.gridsize.z - 1;

  for(size_t z = 0; z <= zmax; ++z) {
    size_t location;

    if(morton_order) {
      location = mortonEncode_LUT(x, y, z);
    } else {
      location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
    }

    if (checkBit(vtable_obj, location)) {
      break;
    }


    unsetBit(vtable_carved, location);
  }

  for (size_t z = zmax; z >= 0; --z) {

    size_t location;

    if(morton_order) {
      location = mortonEncode_LUT(x, y, z);
    } else {
      location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
    }

    if (checkBit(vtable_obj, location)) {
      break;
    }

    unsetBit(vtable_carved, location);
  }
}

__global__ void mergeVtables(unsigned int* v1, unsigned int* v2, const voxinfo& info, bool morton_order) {
  int x = threadIdx.x;
  printf("%d\n", x);

  size_t zmax = info.gridsize.z - 1;
  size_t ymax = info.gridsize.y - 1;

  for (size_t y = 0; y <= ymax; ++y) {

    for (size_t z = 0; z <= zmax; ++z) {

      size_t location;
      if(morton_order) {
        location = mortonEncode_LUT(x, y, z);
      } else {
        location = x + (y * info.gridsize.y) + (z * info.gridsize.y * info.gridsize.z);
      }

      printf("working");
      if (!checkBit(v2, location)) {
        continue;
      }

      setBit(v1, location);
    }
  }
}

void solid_voxelize(const voxinfo& v, std::vector<float*> triangle_data, unsigned int* vtable, 
    bool morton_code) {
  float elapsed_time;


  if (morton_code) {
    checkCudaErrors(cudaMemcpyToSymbol(morton256_x, host_morton256_x, 256 * sizeof(uint32_t)));
    checkCudaErrors(cudaMemcpyToSymbol(morton256_y, host_morton256_y, 256 * sizeof(uint32_t)));
    checkCudaErrors(cudaMemcpyToSymbol(morton256_z, host_morton256_z, 256 * sizeof(uint32_t)));
  }

  int blockSize;
  int minGridSize;
  int gridSize;

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxelize_triangle, 0, 0);
  gridSize = (v.n_triangles + blockSize - 1) / blockSize;
  std::cout << "CUDA GRIDSIZE: " << gridSize << " , " << "BLOCKSIZE: " << blockSize << std::endl;

  size_t vtable_size = ((size_t) v.gridsize.x * v.gridsize.y * v.gridsize.z) / 8.0f;

  unsigned int* vtable_carved;
  checkCudaErrors(cudaMallocManaged((void**)&vtable_carved, vtable_size));
  unsigned int* vtable_obj;
  checkCudaErrors(cudaMallocManaged((void**)&vtable_obj, vtable_size));

  int count = 1;
  for (float* gpu_triangles: triangle_data) {
    std::cout << " - voxelizing obj #  " << count << std::endl;
//    checkCudaErrors(cudaMemset((void**)vtable_obj, -1, vtable_size));
//    checkCudaErrors(cudaMemset((void**)vtable_obj, -1, vtable_size));

    voxelize_triangle<<<gridSize, blockSize>>>(v, gpu_triangles, vtable_obj, morton_code);
    //carveX<<<v.gridsize.y, v.gridsize.z>>>(vtable_obj, vtable_carved, v, morton_code);
    //carveY<<<v.gridsize.x, v.gridsize.z>>>(vtable_obj, vtable_carved, v, morton_code);
    //carveZ<<<v.gridsize.x, v.gridsize.y>>>(vtable_obj, vtable_carved, v, morton_code);
    //mergeVtables<<<v.gridsize.x, v.gridsize.y>>>(vtable_obj, vtable_carved, v, morton_code);
    mergeVtables<<<1, 256>>>(vtable, vtable_obj, v, morton_code);
    cudaDeviceSynchronize();
    ++count;
  }
  // checkCudaErrors(cudaMemcpy(vtable, vtable_obj, vtable_size, cudaMemcpyDeviceToDevice));
 

  //vtable = vtable_obj;
}



void voxelize(const voxinfo& v, float* triangle_data, unsigned int* vtable, 
    bool useThrustPath, bool morton_code, bool solid) {
	float   elapsedTime;

	// These are only used when we're not using UNIFIED memory
	unsigned int* dev_vtable; // DEVICE pointer to voxel_data
	size_t vtable_size; // vtable size
	
	// Create timers, set start time
	cudaEvent_t start_vox, stop_vox;
	checkCudaErrors(cudaEventCreate(&start_vox));
	checkCudaErrors(cudaEventCreate(&stop_vox));

	// Copy morton LUT if we're encoding to morton
	if (morton_code){
		checkCudaErrors(cudaMemcpyToSymbol(morton256_x, host_morton256_x, 256 * sizeof(uint32_t)));
		checkCudaErrors(cudaMemcpyToSymbol(morton256_y, host_morton256_y, 256 * sizeof(uint32_t)));
		checkCudaErrors(cudaMemcpyToSymbol(morton256_z, host_morton256_z, 256 * sizeof(uint32_t)));
	}

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, voxelize_triangle, 0, 0);
	// Round up according to array size 
	gridSize = (v.n_triangles + blockSize - 1) / blockSize;

	if (useThrustPath) { // We're not using UNIFIED memory
		vtable_size = ((size_t)v.gridsize.x * v.gridsize.y * v.gridsize.z) / (size_t) 8.0;
		fprintf(stdout, "[Voxel Grid] Allocating %llu kB of DEVICE memory\n", size_t(vtable_size / 1024.0f));
		checkCudaErrors(cudaMalloc(&dev_vtable, vtable_size));
		checkCudaErrors(cudaMemset(dev_vtable, 0, vtable_size));
		// Start voxelization
		checkCudaErrors(cudaEventRecord(start_vox, 0));
		voxelize_triangle << <gridSize, blockSize >> > (v, triangle_data, dev_vtable, morton_code);
	}
	else { // UNIFIED MEMORY 
		checkCudaErrors(cudaEventRecord(start_vox, 0));
		voxelize_triangle << <gridSize, blockSize >> > (v, triangle_data, vtable, morton_code);
	}

	cudaDeviceSynchronize();
	checkCudaErrors(cudaEventRecord(stop_vox, 0));
	checkCudaErrors(cudaEventSynchronize(stop_vox));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_vox, stop_vox));
	printf("[Voxelization] GPU time:  %3.1f ms\n", elapsedTime);

	// If we're not using UNIFIED memory, copy the voxel table back and free all
	if (useThrustPath){
		fprintf(stdout, "[Voxel Grid] Copying %llu kB to page-locked HOST memory\n", size_t(vtable_size / 1024.0f));
		checkCudaErrors(cudaMemcpy((void*)vtable, dev_vtable, vtable_size, cudaMemcpyDefault));
		fprintf(stdout, "[Voxel Grid] Freeing %llu kB of DEVICE memory\n", size_t(vtable_size / 1024.0f));
		checkCudaErrors(cudaFree(dev_vtable));
	}

	// SANITY CHECKS
	//size_t t_seen, v_count;
	//HANDLE_CUDA_ERROR(cudaMemcpyFromSymbol((void*)&(t_seen),triangles_seen_count, sizeof(t_seen), 0, cudaMemcpyDeviceToHost));
	//HANDLE_CUDA_ERROR(cudaMemcpyFromSymbol((void*)&(v_count), voxel_count, sizeof(v_count), 0, cudaMemcpyDeviceToHost));
	//printf("We've seen %llu triangles on the GPU \n", t_seen);
	//printf("We've found %llu voxels on the GPU \n", v_count);

	// Destroy timers
	checkCudaErrors(cudaEventDestroy(start_vox));
	checkCudaErrors(cudaEventDestroy(stop_vox));
}
