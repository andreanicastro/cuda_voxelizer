#include <iostream>
#include <string>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "util.h"
#include "util_cuda.h"

std::string version_number = "v0.3";

enum OutputFormat { output_binvox, output_morton};
std::string OutputFormats[] = { "binvox file", "morton encoded blob" };

// Default options
std::string filename = "";
std::string filename_base = "";
OutputFormat outputformat = output_binvox;
unsigned int gridsize = 256;
bool useThrustPath = false;

void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

// METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const trimesh::TriMesh *mesh) {
	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
	float* device_triangles;
	fprintf(stdout, "[Mesh] Allocating %llu kB of CUDA-managed UNIFIED memory \n", (size_t)(n_floats / 1024.0f));
	checkCudaErrors(cudaMallocManaged((void**) &device_triangles, n_floats)); // managed memory
	fprintf(stdout, "[Mesh] Copy %llu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->faces.size()));
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		size_t j = i * 9;
		memcpy((device_triangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
		memcpy((device_triangles)+j+3, glm::value_ptr(v1), sizeof(glm::vec3));
		memcpy((device_triangles)+j+6, glm::value_ptr(v2), sizeof(glm::vec3));
	}
	return device_triangles;
}

void printHeader(){
  std::cout << "CUDA Voxelizer " << version_number << " by Jeroen Baert" << std::endl; 
	std::cout << "github.com/Forceflow/cuda_voxelizer - jeroen.baert@cs.kuleuven.be" << std::endl;
}

void printExample() {
  std::cout << "Example: cuda_voxelizer -f <path_to_model> -s 512" << std::endl;
}

void printHelp(){
	fprintf(stdout, "\n## HELP  \n");
	std::cout << "Program options: " << std::endl;
	std::cout << " -f <path to model file: .ply, .obj, .3ds> (required)" << std::endl;
	std::cout << " -s <voxelization grid size, power of 2: 8 -> 512, 1024, ... (default: 256)>" << std::endl;
	std::cout << " -o <output format: binvox or morton (default: binvox)>" << std::endl;
	std::cout << " -t : Force using CUDA Thrust Library (possible speedup / throughput improvement)" << std::endl;
	printExample();
}

// Parse the program parameters and set them as global variables
void parseProgramParameters(int argc, char* argv[]){
	if(argc<2){ // not enough arguments
		fprintf(stdout, "Not enough program parameters. \n \n");
		printHelp();
		exit(0);
	} 
	bool filegiven = false;
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) == "-f") {
			filename = argv[i + 1];
			filename_base = filename.substr(0, filename.find_last_of("."));
			filegiven = true;
			i++;
		}
		else if (std::string(argv[i]) == "-s") {
			gridsize = atoi(argv[i + 1]);
			i++;
		} else if (std::string(argv[i]) == "-h") {
			printHelp();
			exit(0);
		} else if (std::string(argv[i]) == "-o") {
			std::string output = (argv[i + 1]);
			transform(output.begin(), output.end(), output.begin(), ::tolower); // to lowercase
			if (output == "binvox"){
				outputformat = output_binvox;
			}
			else if (output == "morton"){
				outputformat = output_morton;
			}
			else {
				fprintf(stdout, "Unrecognized output format: %s, valid options are binvox (default) or morton \n", 
            output.c_str());
				exit(0);
			}
		}
		else if (std::string(argv[i]) == "-t") {
			useThrustPath = true;
		}
	}
	if (!filegiven) {
		fprintf(stdout, "You didn't specify a file using -f (path). This is required. Exiting. \n");
		printExample();
		exit(0);
	}
	fprintf(stdout, "Filename: %s \n", filename.c_str());
	fprintf(stdout, "Grid size: %i \n", gridsize);
	fprintf(stdout, "Output format: %s \n", OutputFormats[outputformat].c_str());
	fprintf(stdout, "Using CUDA Thrust: %s \n", useThrustPath ? "Yes" : "No");
}


int main(int argc, char *argv[]) {
  printHeader();
  std::cout << "\n## PROGRAM PARAMETERS" << std::endl;
  parseProgramParameters(argc, argv);
  std::cout << "\n## CUDA INIT" << std::endl;
  initCuda();

  trimesh::TriMesh::set_verbose(false);

#ifdef _DEBUG
  std::cout << "\n## MESH IMPORT\n" << std::endl;
  trimesh::TriMesh::set_verbose(true);
#endif
  
  std::cout << "\n Read input mesh" << std::endl;
  std::cout << "\n [I/O] Reading mesh from " << filename << std::endl;
  trimesh::TriMesh *themesh = trimesh::TriMesh::read(filename.c_str());
  std::cout << "[Mesh] Computing faces" << std::endl;
  themesh->need_faces();
  std::cout << "[Mesh] Computing bbox " << std::endl;
  themesh->need_bbox(); // todo will be removed

  // modify for multiple meshes
  float * device_triangles = meshToGPU_managed(themesh);

  std::cout << "\n## VOXELIZATION SETUP" << std::endl;
  AABox<glm::vec3> bbox_mesh(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max));

  voxinfo v(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gridsize, gridsize, gridsize), themesh->faces.size());
  v.print();

  size_t vtable_size = ((size_t) gridsize * gridsize * gridsize) / 8.0f;

  unsigned int* vtable;

  std::cout << "[Voxel Grid] Allocating " << size_t(vtable_size / 1024.0f) << " kB of CUDA-managed UNIFIED memory" 
    << std::endl;

  checkCudaErrors(cudaMallocManaged((void **)&vtable, vtable_size));

  std::cout << "\n## GPU VOXELIZATION" << std::endl;
  voxelize(v, device_triangles, vtable, false, (outputformat == output_morton));
}
