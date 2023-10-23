#include <stdio.h>
#include <stdbool.h>

#include "encode.h"

#define BLOCK_SIZE 256 // Assuming 4x4x4x4 block size


// Test function for gather_block
void test_gather_block() {
    double raw[4][4][4][4]; // 4x4x4x4 source array
    double block[BLOCK_SIZE];

    // Initialize raw with some data (e.g., sequence from 1 to 256)
    for (int w = 0; w < 4; w++)
        for (int z = 0; z < 4; z++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    raw[w][z][y][x] = (double)(x + 4 * y + 16 * z + 64 * w + 1);

    // Test gather_block
    printf("Testing gather_block:\n");
    gather_block(block, (const double*)raw, 1, 4, 16, 64);

    // Check correctness of gathered values
    bool success = true;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (block[i] != (double)(i + 1)) {
            printf("Mismatch at index %d: Expected %f, Got %f\n", i, (double)(i + 1), block[i]);
            success = false;
        }
    }

    if (success) {
        printf("\t TEST PASSED.\n");
    }
}

// Test function for encode_block_strided
void test_encode_block_strided() {
    double raw[4][4][4][4]; // 4x4x4x4 source array
    int encoded[BLOCK_SIZE];

    // Initialize raw with some data (e.g., sequence from 1 to 256)
    for (int w = 0; w < 4; w++)
        for (int z = 0; z < 4; z++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    raw[w][z][y][x] = (double)(x + 4 * y + 16 * z + 64 * w + 1);

    // Test encode_block_strided
    printf("Testing encode_block_strided:\n");
    encode_block_strided(encoded, (const double*)raw, 1, 4, 16, 64);

    // Check correctness of encoded values
    bool success = true;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (encoded[i] != (int)(i + 1)) {
            printf("Mismatch at index %d: Expected %d, Got %d\n", i, (int)(i + 1), encoded[i]);
            success = false;
        }
    }

    if (success) {
        printf("\t TEST PASSED.\n");
    }
}

int main() {
    test_gather_block();
    test_encode_block_strided();

    return 0;
}
