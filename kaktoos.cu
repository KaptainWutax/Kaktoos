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

#undef JRAND_DOUBLE

#define RANDOM_MULTIPLIER_LONG 0x5DEECE66DULL

#ifdef JRAND_DOUBLE
#define Random double
#define RANDOM_MULTIPLIER 0x5DEECE66Dp-48
#define RANDOM_ADDEND 0xBp-48
#define RANDOM_SCALE 0x1p-48

inline uint32_t random_next(Random *random, int32_t bits) {
    *random = trunc((*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) * RANDOM_SCALE);
    return (uint32_t)((uint64_t)(*random / RANDOM_SCALE) >> (48 - bits));
}

#else

#define Random uint64_t
#define RANDOM_MULTIPLIER RANDOM_MULTIPLIER_LONG
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48) - 1)
#define RANDOM_SCALE 1

#define FAST_NEXT_INT

// Random::next(bits)
__host__ __device__ inline uint32_t random_next(Random *random, int32_t bits) {
    *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
    return (uint32_t)(*random >> (48 - bits));
}
#endif // ~JRAND_DOUBLE

// new Random(seed)
#define get_random(seed) ((Random)((seed ^ RANDOM_MULTIPLIER_LONG) & RANDOM_MASK))
#define get_random_unseeded(state) ((Random) ((state) * RANDOM_SCALE))

__host__ __device__ int32_t next_int_unknown(uint64_t *seed, int16_t bound) {
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
__host__ __device__ inline uint32_t random_next_int(Random *random, uint32_t bound) {
    int r = random_next(random, 31);
    int m = bound - 1;
    if ((bound & m) == 0) {
        r = (uint32_t)((bound * (uint64_t)r) >> 31);
    } else {
#ifdef FAST_NEXT_INT
        r %= bound;
#else
        for (int32_t u = r;
            u - (r = u % bound) + m < 0;
            u = random_next(random, 31));
#endif
    }
    return r;
}


#define TOTAL_WORK_SIZE 1LL << 48

#define WORK_UNIT_SIZE (1LL << 20)
#define BLOCK_SIZE 256

__global__ void crack(uint64_t seed_offset, int32_t *num_seeds, uint64_t *seeds) {
    uint64_t originalSeed = blockIdx.x * blockDim.x + threadIdx.x + seed_offset;
    if (originalSeed >= TOTAL_WORK_SIZE)
        return;
    uint64_t seed = originalSeed;
    
    int16_t wantedCactusHeight = 8;
    int16_t floorLevel = 63;
    int16_t attemptsCount = 10;
    int16_t heightMap[1024];
    
    for (int32_t temp = 0; temp < 1024; temp++) {
        heightMap[temp] = floorLevel;
    }
    
    int16_t currentHighestPos = 0;
    int16_t terrainHeight;
    int16_t initialPosX, initialPosY, initialPosZ;
    int16_t posX, posY, posZ;
    int16_t offset, posMap;
    
    int16_t i, a, j;
    
    for (i = 0; i < attemptsCount; i++) {
        if (wantedCactusHeight - heightMap[currentHighestPos] + floorLevel > 9 * (attemptsCount - i))
            return;
            
        initialPosX = random_next(&seed, 4) + 8;
        initialPosZ = random_next(&seed, 4) + 8;
        terrainHeight = (heightMap[initialPosX + initialPosZ * 32] + 1) * 2;
        
        initialPosY = next_int_unknown(&seed, terrainHeight);
        
        for (a = 0; a < 10; a++) {
            posX = initialPosX + random_next(&seed, 3) - random_next(&seed, 3);
            posY = initialPosY + random_next(&seed, 2) - random_next(&seed, 2);
            posZ = initialPosZ + random_next(&seed, 3) - random_next(&seed, 3);
            
            if (posY <= heightMap[posX + posZ * 32] && posY >= 0)
                continue;
            
            offset = 1 + next_int_unknown(&seed, random_next_int(&seed, 3) + 1);
            posMap = posX + posZ * 32;
            
            for (j = 0; j < offset; j++) {
                if ((posY + j - 1) > heightMap[posX + posZ * 32] || posY < 0) continue;
                if((posY + j) <= heightMap[(posX + 1) + posZ * 32] && posY >= 0)continue;
                if((posY + j) <= heightMap[(posX - 1) + posZ * 32] && posY >= 0)continue;
                if((posY + j) <= heightMap[posX + (posZ + 1) * 32] && posY >= 0)continue;
                if((posY + j) <= heightMap[posX + (posZ - 1) * 32] && posY >= 0)continue;
                
                heightMap[posMap]++;
                
                if (heightMap[currentHighestPos] < heightMap[posMap]) {
                    currentHighestPos = posMap;
                }
            }
        }
        
        if (heightMap[currentHighestPos] - floorLevel >= wantedCactusHeight) {
            int32_t index = atomicAdd(num_seeds, 1);
            seeds[index] = originalSeed;
            return;
        }
    }
}

#define GPU_COUNT 1

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
int main() {
    printf("Searching %lld total seeds...\n", TOTAL_WORK_SIZE);

    FILE* out_file = fopen("kaktoos_seeds.txt", "w");

    for(int32_t i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i],i);
    }


    uint64_t count = 0;
    time_t startTime = time(NULL), currentTime;
    for (uint64_t offset = 0; offset < TOTAL_WORK_SIZE;) {

        for(int32_t gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            cudaSetDevice(gpu_index);
            *nodes[gpu_index].num_seeds = 0;
            crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE>>> (offset, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds);
            offset += WORK_UNIT_SIZE;
        }

        for(int32_t gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++) {
            cudaSetDevice(gpu_index);
            cudaDeviceSynchronize();

            for (int32_t i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
                fprintf(out_file, "%lld\n", nodes[gpu_index].seeds[i]);
            }
            fflush(out_file);
            count += *nodes[gpu_index].num_seeds;
        }

        time(&currentTime);
        int timeElapsed = (int)(currentTime - startTime);
        uint64_t numSearched = offset + WORK_UNIT_SIZE;
        double speed = (double)numSearched / (double)timeElapsed / 1000000.0;
        printf("Searched %lld seeds, found %lld matches . Time elapsed: %ds. Speed: %.2fm seeds/s.\n", numSearched, count, timeElapsed, speed);
    }

    fclose(out_file);

}
