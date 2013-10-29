INSTALL := $(HOME)/qrc/track/opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc -lstdc++

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

main: dft

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./dft

clean:
	rm -rf dft *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./dft

.PHONY: main test clean debug
