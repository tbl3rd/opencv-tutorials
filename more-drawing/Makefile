INSTALL := ../opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc -lstdc++

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

main: moreDrawing

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./moreDrawing

clean:
	rm -rf moreDrawing *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./moreDrawing

.PHONY: main test clean debug
