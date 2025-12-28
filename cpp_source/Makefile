CC := g++
CFLAGS := -O2 -std=c++11
SRCS := cpp_kernel.cpp main.cpp
TARGET := conv2d_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
