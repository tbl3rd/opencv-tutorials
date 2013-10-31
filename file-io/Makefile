INSTALL := $(HOME)/qrc/track/opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc -lstdc++

CXXFLAGS := -Wno-c++11-extensions
CXXFLAGS := -g -O0 
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

main: fileIo

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./fileIo

somedata.xml.gz: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./fileIo ./somedata.xml.gz

somedata.yaml.gz: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./fileIo ./somedata.yaml.gz

test: somedata.xml.gz somedata.yaml.gz

clean:
	rm -rf fileIo *.dSYM somedata.xml.gz somedata.yaml.gz

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./fileIo -- somedata.yaml

.PHONY: main help test clean debug
