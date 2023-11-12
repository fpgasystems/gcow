.PHONY: all test clean format encoder_test zfp_test

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
test:
	@echo $(SRC_FILES)
TEST_FILES = $(wildcard $(TEST_DIR)/*.c)
TEST_ENCODE = $(TEST_DIR)/test_encode.cpp
TEST_ZFP = $(TEST_DIR)/test_zfp.cpp

#* Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_FILES))
TEST_ENCODE_OBJS = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_ENCODE))
TEST_ZFP_OBJS = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_ZFP))

#* Header files
INCLUDES = -I$(INCLUDE_DIR)

#* Libraries
LIBRARIES = $(wildcard $(LIB_DIR)/*.a) -lpthread

#* Executable
# EXECUTABLE = test_encoder
EXE_TEST_ENCODER = test_encoder
EXE_TEST_ZFP = test_zfp

#* Main target
all: $(TEST_ENCODER) $(TEST_ZFP) #$(EXECUTABLE)

#* Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

#* Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBRARIES) -c $< -o $@

#* Link the test executable
# $(EXECUTABLE): $(OBJ_FILES) $(TEST_OBJ_FILES)
# 	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LIBRARIES) -o $(BUILD_DIR)/$@

$(EXE_TEST_ENCODER): $(OBJ_FILES) $(TEST_ENCODE_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LIBRARIES) -o $(BUILD_DIR)/$@

$(EXE_TEST_ZFP): $(OBJ_FILES) $(TEST_ZFP_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LIBRARIES) -o $(BUILD_DIR)/$@

#* Run the tests
# test: $(EXECUTABLE)
# 	./$(BUILD_DIR)/$(EXECUTABLE)

encoder_test: $(EXE_TEST_ENCODER)
	./$(BUILD_DIR)/$(EXE_TEST_ENCODER)

zfp_test: $(EXE_TEST_ZFP)
	./$(BUILD_DIR)/$(EXE_TEST_ZFP)

#* Clean build files
clean:
	rm -rf $(BUILD_DIR) **/*.orig tests/data/*.gcow

format:clean
	astyle --style=linux --indent=spaces=2 -S --max-code-length=80 src/*.c* include/*.h* tests/*.c*
