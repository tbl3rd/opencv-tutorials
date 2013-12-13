INSTALL := ../opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := blink
ARGS := 500 ./left.png ./right.png

main: $(EXECUTABLE)

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE)

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(ARGS)

clean:
	rm -rf $(EXECUTABLE) gray-image.jpg *.dSYM

debug: $(EXECUTABLE)
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- $(ARGS)


.PHONY: main test clean debug
