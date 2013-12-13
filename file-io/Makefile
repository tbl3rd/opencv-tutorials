INSTALL := ../opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc -lstdc++

CXXFLAGS := -Wno-c++11-extensions
CXXFLAGS := -g -O0 
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := fileIo

main: $(EXECUTABLE)

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE)

somedata.xml.gz: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) ./somedata.xml.gz

somedata.yaml.gz: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) ./somedata.yaml.gz

test: somedata.xml.gz somedata.yaml.gz

clean:
	rm -rf $(EXECUTABLE) *.dSYM somedata.xml.gz somedata.yaml.gz

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- somedata.yaml

.PHONY: main help test clean debug
