#include <stdio.h>
#include <stdbool.h>

#include "gtest/gtest.h"

// extern "C" {
//    // * Avoid name mangling for C functions.
#include "encode.h"
// }

#define BLOCK_SIZE 256 // Assuming 4x4x4x4 block size


// Test function for gather_block
TEST(encode, gather_block) {
    printf("\n");
    double raw[4][4][4][4]; // 4x4x4x4 source array
    double block[BLOCK_SIZE];

    // Initialize raw with some data (e.g., sequence from 1 to 256)
    for (int w = 0; w < 4; w++)
        for (int z = 0; z < 4; z++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    raw[w][z][y][x] = (double)(x + 4 * y + 16 * z + 64 * w + 1);

    // Test gather_block
    gather_block(block, (const double*)raw, 1, 4, 16, 64);

    // Check correctness of gathered values
    bool success = true;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        EXPECT_EQ(block[i], (double)(i + 1));
    }
}

// Test function for encode_block_strided
TEST(encode, encode_block_strided) {
    printf("\n");
    double raw[4][4][4][4]; // 4x4x4x4 source array
    int encoded[BLOCK_SIZE];

    // Initialize raw with some data (e.g., sequence from 1 to 256)
    for (int w = 0; w < 4; w++)
        for (int z = 0; z < 4; z++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    raw[w][z][y][x] = (double)(x + 4 * y + 16 * z + 64 * w + 1);

    // Test encode_block_strided
    encode_block_strided(encoded, (const double*)raw, 1, 4, 16, 64);

    // Check correctness of encoded values
    bool success = true;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        EXPECT_EQ(encoded[i], (int)(i + 1));
    }
}

int main(int argc, char** argv) {
    printf("\n");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

