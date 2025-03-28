NVCC = nvcc
TARGET = main
SRC_DIR = src

NVCC_FLAGS = --std=c++11 -Wno-deprecated-gpu-targets
INCLUDE_DIRS = -I$(SRC_DIR)

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -o $@ $<

clean:
	rm -f $(TARGET)
