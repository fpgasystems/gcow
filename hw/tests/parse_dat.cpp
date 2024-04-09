#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // Open the file in binary mode
    std::ifstream file("./tests/data/AEROD_v_1_1800_3600.dat", std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file." << std::endl;
        return 1;
    }

    // Read floats from the file
    const int rows = 1800;
    const int cols = 3600;
    std::vector<float> data(rows * cols);

    // Read data byte by byte and convert from little-endian to native format
    for (int i = 0; i < rows * cols; ++i) {
        uint32_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        data[i] = *reinterpret_cast<float*>(&value);
    }

    // Close the file
    file.close();

    std::cout << "Length of data: " << data.size() << std::endl;
    std::cout << "First 10 elements:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << std::endl;
    }
    // Now you have the data stored in the vector 'data'
    // You can use it as needed

    return 0;
}
