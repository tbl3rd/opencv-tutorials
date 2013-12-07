INSTALL := $(HOME)/qrc/track/opencv/install
LIBS := \
-lopencv_calib3d \
-lopencv_core \
-lopencv_flann \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_features2d \
-lopencv_nonfree \
-lstdc++ \
#

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := homographyPerspective
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
