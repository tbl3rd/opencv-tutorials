INSTALL := $(HOME)/qrc/track/opencv/install
LIBS := \
-lopencv_core \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_objdetect \
-lstdc++ \
#

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := cascadeClassify

main: $(EXECUTABLE)

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH ./$(EXECUTABLE)

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(IMAGEFILES)

clean:
	rm -rf $(EXECUTABLE) *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- $(IMAGEFILES)

.PHONY: main help test clean debug
