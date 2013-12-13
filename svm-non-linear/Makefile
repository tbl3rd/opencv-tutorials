INSTALL := ../opencv/install
LIBS := \
-lopencv_core \
-lopencv_highgui \
-lopencv_ml \
-lstdc++ \
#

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := old-svmNonlinear
EXECUTABLE := svmNonlinear

main: $(EXECUTABLE)

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH ./$(EXECUTABLE)

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(ARGS)

clean:
	rm -rf $(EXECUTABLE) *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- $(ARGS)

.PHONY: main help test clean debug
