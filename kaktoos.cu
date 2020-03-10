// IDE indexing
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#define __CUDACC__
#include <device_functions.h>
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>
#endif

#include <stdint.h>
#include <memory.h>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>

#define RANDOM_MULTIPLIER 0x5DEECE66DULL
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48) - 1)
#define RANDOM_SCALE 1

#ifndef FLOOR_LEVEL
#define FLOOR_LEVEL 63
#endif

#ifndef WANTED_CACTUS_HEIGHT
#define WANTED_CACTUS_HEIGHT 8
#endif

// Random::next(bits)
__device__ inline uint32_t random_next(uint64_t *random, int32_t bits) {
    *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
    return (uint32_t)(*random >> (48 - bits));
}

// new Random(seed)
#define get_random(seed) ((uint64_t)((seed ^ RANDOM_MULTIPLIER_LONG) & RANDOM_MASK))
#define get_random_unseeded(state) ((uint64_t) ((state) * RANDOM_SCALE))

__device__ int32_t next_int_unknown(uint64_t *seed, int16_t bound) {
    if ((bound & -bound) == bound) {
        *seed = (*seed * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
        return (int32_t)((bound * (*seed >> 17)) >> 31);
    }

    int32_t bits, value;
    do {
        *seed = (*seed * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
        bits = *seed >> 17;
        value = bits % bound;
    } while (bits - value + (bound - 1) < 0);
    return value;
}

// Random::nextInt(bound)
__device__ inline uint32_t random_next_int(uint64_t *random) {
    return random_next(random, 31) % 3;
}

#define TOTAL_WORK_SIZE (1LL << 48)

#ifndef WORK_UNIT_SIZE
#define WORK_UNIT_SIZE (1LL << 23)
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__device__ inline int8_t extract(int32_t heightMap[], int32_t i) {
    return (int8_t)(heightMap[i >> 2] >> ((i & 0b11) << 3));
}

__device__ inline void increase(int32_t heightMap[], int32_t i) {
    heightMap[i >> 2] += 1 << ((i & 0b11) << 3);
}

__global__ void crack(uint64_t seed_offset, int32_t *num_seeds, uint64_t *seeds) {
    uint64_t originalSeed = blockIdx.x * blockDim.x + threadIdx.x + seed_offset;
    uint64_t seed = originalSeed;

    int32_t heightMap[256];

    for (int32_t temp = 0; temp < 256; temp++) {
        heightMap[temp] = FLOOR_LEVEL | FLOOR_LEVEL << 8 | FLOOR_LEVEL << 16 | FLOOR_LEVEL << 24;
    }

    int16_t currentHighestPos = 0;
    int16_t terrainHeight;
    int16_t initialPosX, initialPosY, initialPosZ;
    int16_t posX, posY, posZ;
    int16_t offset, posMap;

    int16_t i, a, j;

    for (i = 0; i < 10; i++) {
        // Keep, most threads finish early this way
        if (WANTED_CACTUS_HEIGHT - extract(heightMap, currentHighestPos) + FLOOR_LEVEL > 9 * (10 - i))
            return;

        initialPosX = random_next(&seed, 4) + 8;
        initialPosZ = random_next(&seed, 4) + 8;
        terrainHeight = (extract(heightMap, initialPosX + initialPosZ * 32) + 1) * 2;

        initialPosY = next_int_unknown(&seed, terrainHeight);

        for (a = 0; a < 10; a++) {
            posX = initialPosX + random_next(&seed, 3) - random_next(&seed, 3);
            posY = initialPosY + random_next(&seed, 2) - random_next(&seed, 2);
            posZ = initialPosZ + random_next(&seed, 3) - random_next(&seed, 3);

            posMap = posX + posZ * 32;
            // Keep
            if (posY <= extract(heightMap, posMap) && posY >= 0)
                continue;

            offset = 1 + next_int_unknown(&seed, random_next_int(&seed) + 1);

            for (j = 0; j < offset; j++) {
                if ((posY + j - 1) > extract(heightMap, posMap) || posY < 0) continue;
                if ((posY + j) <= extract(heightMap, (posX + 1) + posZ * 32) && posY >= 0) continue;
                if ((posY + j) <= extract(heightMap, posX + (posZ - 1) * 32) && posY >= 0) continue;
                if ((posY + j) <= extract(heightMap, (posX - 1) + posZ * 32) && posY >= 0) continue;
                if ((posY + j) <= extract(heightMap, posX + (posZ + 1) * 32) && posY >= 0) continue;

                increase(heightMap, posMap);

                if (extract(heightMap, currentHighestPos) < extract(heightMap, posMap)) {
                    currentHighestPos = posMap;
                }
            }
        }

        if (extract(heightMap, currentHighestPos) - FLOOR_LEVEL >= WANTED_CACTUS_HEIGHT) {
            int32_t index = atomicAdd(num_seeds, 1);
            seeds[index] = originalSeed;
            return;
        }
    }
}

#ifndef GPU_COUNT
#define GPU_COUNT 1
#endif

struct GPU_Node {
    int GPU;
    int* num_seeds;
    uint64_t* seeds;
};

void setup_gpu_node(GPU_Node* node, int32_t gpu) {
    cudaSetDevice(gpu);
    node->GPU = gpu;
    cudaMallocManaged(&node->num_seeds, sizeof(*node->num_seeds));
    cudaMallocManaged(&node->seeds, (1LL << 10)); // approx 1kb
}

GPU_Node nodes[GPU_COUNT];
int32_t processed[GPU_COUNT];
uint64_t offset = 0;
uint64_t count = 0;
std::mutex info_lock;

void gpu_manager(int32_t gpu_index) {
    std::string fileName = "kaktoos_seeds" + std::to_string(gpu_index) + ".txt";
    FILE *out_file = fopen(fileName.c_str(), "w");
    cudaSetDevice(gpu_index);
    while (offset < TOTAL_WORK_SIZE) {
        *nodes[gpu_index].num_seeds = 0;
        crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>> (offset, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds);
        info_lock.lock();
        offset += WORK_UNIT_SIZE;
        info_lock.unlock();
        cudaDeviceSynchronize();
        for (int32_t i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
            fprintf(out_file, "%lld, %lld\n", (long long int)nodes[gpu_index].seeds[i], (long long int)offset - WORK_UNIT_SIZE);
        }
        fflush(out_file);
        info_lock.lock();
        count += *nodes[gpu_index].num_seeds;
        info_lock.unlock();
    }
    fclose(out_file);
}

int main() {
    printf("Searching %lld total seeds...\n", TOTAL_WORK_SIZE);

    std::thread threads[GPU_COUNT];

    time_t startTime = time(NULL), currentTime;
    for(int32_t i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i], i);
        threads[i] = std::thread(gpu_manager, i);
    }

    using namespace std::chrono_literals;

    while (offset < TOTAL_WORK_SIZE) {
        time(&currentTime);
        int timeElapsed = (int)(currentTime - startTime);
        double speed = (double)offset / (double)timeElapsed / 1000000.0;
        printf("Searched %lld seeds, found %lld matches. Time elapsed: %ds. Speed: %.2fm seeds/s. %f%%\n",
            (long long int)offset, (long long int)count, timeElapsed, speed, (double)offset / TOTAL_WORK_SIZE * 100);
        std::this_thread::sleep_for(0.5s);
    }

}
