.PHONY: all test clean

#* Compiler and flags
CC = g++
CFLAGS = -std=c++11 -Wall -Wextra -Isrc
# CURRENT_DIR := $(shell pwd)

#* Directories
SRC_DIR = src
LIB_DIR = lib
TEST_DIR = tests
BUILD_DIR = build
INCLUDE_DIR = include

#* Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
TEST_FILES = $(wildcard $(TEST_DIR)/*.c)

#* Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_FILES))

#* Header files
INCLUDES = -I$(INCLUDE_DIR)

#* Libraries
LIBRARIES = $(wildcard $(LIB_DIR)/*.a) -lpthread

#* Executable
EXECUTABLE = test_encoder

#* Main target
all: $(EXECUTABLE)

#* Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

#* Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBRARIES) -c $< -o $@

#* Link the test executable
$(EXECUTABLE): $(OBJ_FILES) $(TEST_OBJ_FILES)
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LIBRARIES) -o $(BUILD_DIR)/$@

#* Run the tests
test: $(EXECUTABLE)
	./$(BUILD_DIR)/$(EXECUTABLE)

#* Clean build files
clean:
	rm -rf $(BUILD_DIR) **/*.orig
