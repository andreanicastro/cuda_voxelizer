#include <iostream>
#include <string>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>

#include "util.h"
#include "util_io.h"
#include "util_cuda.h" 
#include "TriMesh_algo.h"

std::string version_number = "v0.3";

enum OutputFormat { output_binvox, output_morton};
std::string OutputFormats[] = { "binvox file", "morton encoded blob" }; 
std::string YcbVideoClasses[]  = {
  "__background__",
  "002_master_chef_can",
  "003_cracker_box",
  "004_sugar_box",
  "005_tomato_soup_can",
  "006_mustard_bottle",
  "007_tuna_fish_can",
  "008_pudding_box",
  "009_gelatin_box", 
  "010_potted_meat_can",
  "011_banana",
  "019_pitcher_base",
  "021_bleach_cleanser",
  "024_bowl",
  "025_mug",
  "035_power_drill",
  "036_wood_block",
  "037_scissors",
  "040_large_marker", 
  "051_large_clamp",
  "052_extra_large_clamp",
  "061_foam_brick" };


// Default options
std::string model_folder = "";
std::string filename = "";
std::string filename_base = "";
OutputFormat outputformat = output_binvox;
unsigned int gridsize = 256;
float griddim = 1.5;
bool useThrustPath = false;

void solid_voxelize(const voxinfo & v,std::vector<float*>& device_triangles, unsigned int* vtable, 
    bool morton_coded);

size_t computeNumberOfFaces(const std::vector<std::shared_ptr<trimesh::TriMesh>>& meshes) {
  size_t n_faces = 0; 
  for (const auto& mesh: meshes) {
    n_faces += mesh->faces.size();
  }
  return n_faces;
}

float* meshesToGPU(const std::vector<std::shared_ptr<trimesh::TriMesh>>& meshes){
  const size_t n_faces = computeNumberOfFaces(meshes);
  const size_t vertices_in_face = 3;
  const size_t coords_per_vertex = 3;

  const size_t n_floats = sizeof(float) * vertices_in_face * coords_per_vertex * n_faces;

  std::cout << "[Mesh] Allocating " << (size_t)(n_floats / 1024.0f) << " kB of CUDA-manage UNIFIED memory" << std::endl;

  float* device_triangles;
  checkCudaErrors(cudaMallocManaged((void**) &device_triangles, n_floats));
  
  std::cout << "[Mesh] Copy " << n_faces << " trinagles ot CUDA-manages UNIFIED memory" << std::endl;

  size_t mesh_offset = 0;
  for (const auto& mesh: meshes) {
    for (size_t i = 0; i < mesh->faces.size(); ++i) {
      glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
      glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
      glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
      const size_t offset =  mesh_offset + i * vertices_in_face * coords_per_vertex;
      memcpy((device_triangles) + offset                        , glm::value_ptr(v0), sizeof(glm::vec3));
      memcpy((device_triangles) + offset + coords_per_vertex    , glm::value_ptr(v1), sizeof(glm::vec3));
      memcpy((device_triangles) + offset + 2 * coords_per_vertex, glm::value_ptr(v2), sizeof(glm::vec3));
    }
    mesh_offset += mesh->faces.size() * vertices_in_face * coords_per_vertex;
  }
  return device_triangles;
}


void multipleMeshesToGPU(const std::vector<std::shared_ptr<trimesh::TriMesh>>& meshes,
                         std::vector<float*>& device_triangles){
  const size_t vertices_in_face = 3;
  const size_t coords_per_vertex = 3;


  for (int i = 0; i < meshes.size(); ++i) {
    const auto& mesh = meshes[i];
    size_t n_faces = mesh->faces.size();
    const size_t n_floats = sizeof(float) * vertices_in_face * coords_per_vertex * n_faces;
    
    float* t;
    checkCudaErrors(cudaMallocManaged((void**) &t, n_floats));
    for (size_t i = 0; i < n_faces; ++i) {
      glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
      glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
      glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
      const size_t offset =  i * vertices_in_face * coords_per_vertex;
      memcpy((t) + offset                         , glm::value_ptr(v0), sizeof(glm::vec3));
      memcpy((t) + offset + coords_per_vertex     , glm::value_ptr(v1), sizeof(glm::vec3));
      memcpy((t) + offset + 2 * coords_per_vertex , glm::value_ptr(v2), sizeof(glm::vec3));
    }
    device_triangles.push_back(t);
    break;
  }
}


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
  std::cout << "Example: cuda_voxelizer -f <path_to_model> -m <path_to_models> -s 512" << std::endl;
}



void printHelp(){
	fprintf(stdout, "\n## HELP  \n");
	std::cout << "Program options: " << std::endl;
	std::cout << " -f <path to meta file: .json> (required)" << std::endl;
  std::cout << " -m <path to model folder> (required)" << std::endl;
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
  bool modelgiven = false;
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) == "-f") {
			filename = argv[i + 1];
			filename_base = filename.substr(0, filename.find_last_of("."));
			filegiven = true;
			i++;
		}
    if (std::string(argv[i]) == "-m") {
      model_folder = argv[i + 1];
      modelgiven = true;
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
  if (!modelgiven) {
    std::cout << "You didn't specify a folder with the models using -m (path). This is required. Exiting." << std::endl;
    printExample();
    exit(0);
  }
	fprintf(stdout, "Filename: %s \n", filename.c_str());
	fprintf(stdout, "Grid size: %i \n", gridsize);
	fprintf(stdout, "Output format: %s \n", OutputFormats[outputformat].c_str());
	fprintf(stdout, "Using CUDA Thrust: %s \n", useThrustPath ? "Yes" : "No");
}

void loadMeshes(const nlohmann::json& jsonfile, std::vector<std::shared_ptr<trimesh::TriMesh>>& meshes) {
  for (const auto& vals: jsonfile["cls_indexes"]) {
    std::string model_name = YcbVideoClasses[vals[0].get<int>()];
    std::cout << "[I/O] loading mesh for object: " << model_name << std::endl;
    std::string model_path = model_folder + "/" + model_name + "/textured.obj";
    std::shared_ptr<trimesh::TriMesh> meshptr(trimesh::TriMesh::read(model_path.c_str()));
    meshptr->need_faces();
    meshptr->need_bbox();
    meshes.push_back(meshptr);
  }
}

void readCameraPose(const nlohmann::json& jsondata, trimesh::xform& camera_pose) {
  auto rotation_translation_matrix = jsondata["rotation_translation_matrix"];
  for (size_t row = 0; row < rotation_translation_matrix.size(); ++row) {
    for (size_t col = 0; col < rotation_translation_matrix[0].size(); ++col) {
      camera_pose(row, col) = rotation_translation_matrix[row][col].get<double>();
    }
  }
}

void readObjectPoses(const nlohmann::json& jsondata, std::vector<trimesh::xform>& object_poses) {
  size_t num_objects = object_poses.capacity();
  size_t num_rows = jsondata["poses"].size();
  size_t num_cols = jsondata["poses"][0].size();

  std::cout << "---------------" << std::endl;
  std::cout << jsondata["poses"].size() << std::endl;
  std::cout << jsondata["poses"][0].size() << std::endl;
  std::cout << jsondata["poses"][0][0].size() << std::endl;
  std::cout << "---------------" << std::endl;
  for (size_t obj = 0; obj < num_objects; ++obj) {
    trimesh::xform pose;
    for (size_t row = 0; row < num_rows; ++row) {
      for (size_t col = 0; col < num_cols; ++ col) {
        pose(row, col) = jsondata["poses"][row][col][obj].get<double>();
      }
    }
    std::cout << "Pose for the obejct " << obj << "\n" << pose << std::endl;
    object_poses.push_back(pose);
  }
}

void transformMeshes(trimesh::xform& camera_pose, 
                  std::vector<trimesh::xform>& object_poses,
                  std::vector<std::shared_ptr<trimesh::TriMesh>>& meshes) {
  const size_t num_meshes = meshes.size();
  for (size_t obj = 0; obj < num_meshes; ++obj) {
    trimesh::xform inv_camera_pose = trimesh::inv(camera_pose);
    trimesh::xform obj_to_world = inv_camera_pose * object_poses[obj];
 //   std::cout << "obj " << obj << " has transform\n" <<  obj_to_world << std::endl;
    trimesh::apply_xform(meshes[obj].get(), obj_to_world); 
  }
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
  
  // read json metadata 
  
  std::cout << "\n Read input meta file" << std::endl;
  std::ifstream jsonfile(filename);
  nlohmann::json jsondata;
  jsonfile >> jsondata;

  std::cout << "\n[I/O] Reading camera pose " << std::endl;

  trimesh::xform camera_pose;
  readCameraPose(jsondata, camera_pose);

  std::cout << "Camera pose: \n" <<  camera_pose << std::endl;

  std::cout << "\n [I/O] Reading meshes from " << filename << std::endl;

  std::vector<std::shared_ptr<trimesh::TriMesh>> themeshes;
  loadMeshes(jsondata, themeshes);

  std::cout << "Loaded " << themeshes.size() << " meshes from disk" << std::endl;

  std::cout << "\n[I/O] Reading object poses " << std::endl;

  std::vector<trimesh::xform> object_poses;
  object_poses.reserve(themeshes.size());
  readObjectPoses(jsondata, object_poses);


  std::cout << "\n[Meshes] Transforming the meshes" << std::endl;
  transformMeshes(camera_pose, object_poses, themeshes);

  std::cout << "Moving meshes to Device" << std::endl;

  std::vector<float*> device_triangles;
  multipleMeshesToGPU(themeshes, device_triangles);

  std::cout << "\n## VOXELIZATION SETUP" << std::endl;
  std::cout << "\tgrid delimiters:" << std::endl;
    
  glm::vec3 bbox_min(- 0.5 * griddim, - 0.5 * griddim, - 0.2 * griddim);
  glm::vec3 bbox_max(  0.5 * griddim,   0.5 * griddim,   0.8 * griddim);

  const size_t faces = computeNumberOfFaces(themeshes);
  AABox<glm::vec3> bbox_mesh(bbox_min, bbox_max);
  voxinfo vinfo(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gridsize, gridsize, gridsize), faces);
  vinfo.print();

  size_t vtable_size = ((size_t) gridsize * gridsize * gridsize) / 8.0f;

  unsigned int* vtable;
  std::cout << "[Voxel Grid] Allocating " << size_t(vtable_size / 1024.0f) << " kB of CUDA-managed UNIFIED memory" 
    << std::endl;
  checkCudaErrors(cudaMallocManaged((void **)&vtable, vtable_size));

  std::cout << "\n## GPU VOXELIZATION" << std::endl;
  solid_voxelize(vinfo, device_triangles, vtable,  (outputformat == output_morton));
  

  if (outputformat == output_morton) {
    std::cout << "\n## OUTPUT TO BINARY FILE" << std::endl;
    write_binary(vtable, vtable_size, filename);
  }
  else if (outputformat == output_binvox) {
    std::cout << "\n## OUTPUT TO BINVOX FILE" << std::endl;
    std::string binvox_file = filename.substr(0, filename.find_last_of("/")) + "/scene.binvox";
    write_binvox(vtable, binvox_file, vinfo);
  }
}
