INSTALL := $(HOME)/qrc/track/opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc -lstdc++

CXXFLAGS := -Wno-c++11-extensions
CXXFLAGS := -g -O0 
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

IOFILE := ./somedata.gz

main: fileIo

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./fileIo $(IOFILE)

clean:
	rm -rf fileIo $(IOFILE) *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./fileIo -- $(IOFILE)

.PHONY: main test clean debug
