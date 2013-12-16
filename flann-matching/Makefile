INSTALL := ../opencv/install
LIBS := \
-lopencv_core \
-lopencv_flann \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_features2d \
-lopencv_nonfree \
#

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := flannMatcher
IMAGEFILES := ../resources/box.png ../resources/box_in_scene.png

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
