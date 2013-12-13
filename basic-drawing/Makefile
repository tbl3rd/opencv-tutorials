INSTALL := ../opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := basicDrawing

main: $(EXECUTABLE)

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE)

.PHONY: main test clean debug
